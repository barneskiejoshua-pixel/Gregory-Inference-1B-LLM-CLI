/*
 * ternary_matmul.c -- Gregory's fast I2_S matvec (int8 act x 2-bit ternary).
 *
 * bitnet.cpp-style hot path: int8 activations multiplied against 2-bit ternary
 * weights via 256-bit _mm256_maddubs_epi16, accumulating int16 -> int32. No
 * float in the inner loop, no per-weight sign/convert.
 *
 * Packed weights store codes c in {0,1,2} where the real ternary weight is
 * w = c - 1. Therefore:
 *
 *     sum_k w[k]*x[k] = sum_k (c[k]-1)*x[k] = (sum_k c[k]*x[k]) - (sum_k x[k])
 *
 * maddubs computes sum_k c[k]*x[k] directly (c unsigned, x signed int8); the
 * -sum_k x[k] bias is the same for every output row, computed once and
 * subtracted per row. This avoids unpacking to {-1,0,+1} and the sign step.
 *
 * Packing layout (see gregory/kernels/__init__.py pack_ternary): row-major,
 * K/4 bytes per row; within a byte, elements k+0,k+1,k+2,k+3 sit at bit
 * shifts 6,4,2,0.
 *
 * Build:
 *   gcc -O3 -mavx2 -mfma -fopenmp -fno-math-errno -fPIC -shared \
 *       ternary_matmul.c -o ternary_matmul.so
 *
 * NOT -ffast-math: it licenses FP reassociation/flush-to-zero that can break
 * the bit-identical AVX2==scalar invariant across toolchains. The integer dot
 * is exact regardless; -fno-math-errno keeps the quantize loop vectorizable.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GK_HAVE_AVX2 1
#else
#define GK_HAVE_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
/* Thread even the small K/V matvecs (640x2560 = 1.6M); fork-join is amortized
 * across the row work at this size. */
#define GK_OMP_THRESHOLD 262144L
#endif

/* Reusable per-thread int8 activation scratchpad. The decode hot path calls
 * the matvec ~210 times per token (7 projections x 30 layers) plus the head;
 * a malloc/free of the K-length quant buffer on every call is pure allocator
 * churn. A thread-local grow-only buffer amortizes it to one allocation per
 * thread for the largest K ever seen. __thread keeps it safe if ctypes releases
 * the GIL and two Python threads enter concurrently; the quantization that
 * fills it runs in the serial region before any OpenMP fork. */
static __thread int8_t *gk_xq_buf = NULL;
static __thread size_t  gk_xq_cap = 0;

static int8_t *gk_xq_scratch(int K) {
    if ((size_t)K > gk_xq_cap) {
        free(gk_xq_buf);
        gk_xq_buf = (int8_t *)malloc((size_t)K);
        gk_xq_cap = gk_xq_buf ? (size_t)K : 0;
    }
    return gk_xq_buf;
}

int gk_has_avx2(void) { return GK_HAVE_AVX2; }
int gk_has_omp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/* Symmetric absmax int8 quantization of the activation vector. Returns the
 * scale; also outputs the integer sum of the quantized activations (for the
 * +1 code-bias correction). Clamps to [-127,127]. */
static inline float
gk_quantize_x(int K, const float *X, int8_t *Xq, long *sx_sum_out) {
    /* Pass 1: absmax. AVX2 reads 8 lanes/iter and clears the sign bit with an
     * and-mask (no compare). The horizontal max is exact, so this is bit-for-
     * bit the same xmax the scalar reduction produced. */
    float xmax = 0.0f;
    int k = 0;
#if GK_HAVE_AVX2
    {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 vmax = _mm256_setzero_ps();
        for (; k + 8 <= K; k += 8) {
            __m256 v = _mm256_and_ps(_mm256_loadu_ps(X + k), absmask);
            vmax = _mm256_max_ps(vmax, v);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, vmax);
        for (int i = 0; i < 8; i++)
            if (lanes[i] > xmax) xmax = lanes[i];
    }
#endif
    for (; k < K; k++) {
        float a = X[k] < 0.0f ? -X[k] : X[k];
        if (a > xmax) xmax = a;
    }

    if (xmax <= 0.0f) {
        for (int j = 0; j < K; j++) Xq[j] = 0;
        *sx_sum_out = 0;
        return 0.0f;
    }

    /* Pass 2: scale + round-half-away-from-zero + clamp. copysignf(0.5,f) makes
     * the round branchless (and equal to the old f>=0 ? f+0.5 : f-0.5), so the
     * compiler can auto-vectorize the multiply/convert (-fno-math-errno). */
    float inv = 127.0f / xmax;
    long s = 0;
    for (int j = 0; j < K; j++) {
        float f = X[j] * inv;
        int v = (int)(f + copysignf(0.5f, f));
        if (v > 127) v = 127; else if (v < -127) v = -127;
        Xq[j] = (int8_t)v;
        s += v;
    }
    *sx_sum_out = s;
    return xmax / 127.0f;
}

#if GK_HAVE_AVX2
static inline int
gk_reduce_add_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Unpack 8 packed bytes -> 32 int8 codes in {0,1,2}. Byte bits
 * [7:6],[5:4],[3:2],[1:0] map to elements k+0,k+1,k+2,k+3. */
static inline void
gk_unpack32_codes(const uint8_t *src, __m128i *out_lo, __m128i *out_hi) {
    __m128i bytes = _mm_loadl_epi64((const __m128i *)src);
    __m128i mask  = _mm_set1_epi8(0x03);
    __m128i c6 = _mm_and_si128(_mm_srli_epi16(bytes, 6), mask);
    __m128i c4 = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m128i c2 = _mm_and_si128(_mm_srli_epi16(bytes, 2), mask);
    __m128i c0 = _mm_and_si128(bytes, mask);
    __m128i p64 = _mm_unpacklo_epi8(c6, c4);
    __m128i p20 = _mm_unpacklo_epi8(c2, c0);
    *out_lo = _mm_unpacklo_epi16(p64, p20);  /* elems 0..15 */
    *out_hi = _mm_unpackhi_epi16(p64, p20);  /* elems 16..31 */
}

/* int dot of one packed ternary row against the quantized activation. The
 * scalar tail only runs when K is not a multiple of 32 (never for real BitNet
 * tensors, where K in {640,2560,6912}); kept for arbitrary-K correctness. */
static inline long
gk_ternary_row_dot(const uint8_t *row, const int8_t *Xq, int K, int K32,
                   __m256i ones16) {
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < K32; k += 32) {
        __m128i ca, cb;
        gk_unpack32_codes(row + (k / 4), &ca, &cb);
        __m256i codes = _mm256_set_m128i(cb, ca);
        __m256i xq    = _mm256_loadu_si256((const __m256i *)(Xq + k));
        __m256i p16   = _mm256_maddubs_epi16(codes, xq);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones16));
    }
    long dot = gk_reduce_add_epi32(acc);
    for (int k = K32; k < K; k += 4) {
        uint8_t b = row[k / 4];
        dot += (long)((b >> 6) & 0x3) * (long)Xq[k + 0];
        if (k + 1 < K) dot += (long)((b >> 4) & 0x3) * (long)Xq[k + 1];
        if (k + 2 < K) dot += (long)((b >> 2) & 0x3) * (long)Xq[k + 2];
        if (k + 3 < K) dot += (long)((b >> 0) & 0x3) * (long)Xq[k + 3];
    }
    return dot;
}

static void
gk_matvec_avx2(int M, int K, const uint8_t *W_packed, float w_scale,
               const float *X, float *Y) {
    int K32 = K & ~31;
    int rowb = K / 4;
    int8_t *Xq = gk_xq_scratch(K);
    if (Xq == NULL) {              /* scratch alloc failed: defined zero output */
        for (int m = 0; m < M; m++) Y[m] = 0.0f;
        return;
    }
    long sx_sum;
    float x_scale = gk_quantize_x(K, X, Xq, &sx_sum);
    if (x_scale == 0.0f) {
        for (int m = 0; m < M; m++) Y[m] = 0.0f;
        return;
    }
    float out_scale = x_scale * w_scale;
    const __m256i ones16 = _mm256_set1_epi16(1);
    int M4 = M & ~3;

    /* Block the output rows by 4 so a single 32-wide load of the shared
     * activation `xq` feeds four weight rows (GEMV: the activation vector is
     * reused across every row). Cuts L1 read traffic and loop overhead ~4x.
     * Parallelized over 4-row blocks; M is a multiple of 4 for every real
     * tensor, so the <4 remainder below is just a correctness backstop. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        if((long)M * K >= GK_OMP_THRESHOLD)
    #endif
    for (int m = 0; m < M4; m += 4) {
        const uint8_t *r0 = W_packed + (size_t)(m + 0) * rowb;
        const uint8_t *r1 = W_packed + (size_t)(m + 1) * rowb;
        const uint8_t *r2 = W_packed + (size_t)(m + 2) * rowb;
        const uint8_t *r3 = W_packed + (size_t)(m + 3) * rowb;
        __m256i a0 = _mm256_setzero_si256();
        __m256i a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256();
        __m256i a3 = _mm256_setzero_si256();

        for (int k = 0; k < K32; k += 32) {
            __m256i xq = _mm256_loadu_si256((const __m256i *)(Xq + k));
            int kb = k / 4;
            __m128i ca, cb;
            gk_unpack32_codes(r0 + kb, &ca, &cb);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_set_m128i(cb, ca), xq), ones16));
            gk_unpack32_codes(r1 + kb, &ca, &cb);
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_set_m128i(cb, ca), xq), ones16));
            gk_unpack32_codes(r2 + kb, &ca, &cb);
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_set_m128i(cb, ca), xq), ones16));
            gk_unpack32_codes(r3 + kb, &ca, &cb);
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_set_m128i(cb, ca), xq), ones16));
        }

        long d0 = gk_reduce_add_epi32(a0);
        long d1 = gk_reduce_add_epi32(a1);
        long d2 = gk_reduce_add_epi32(a2);
        long d3 = gk_reduce_add_epi32(a3);
        for (int k = K32; k < K; k += 4) {           /* scalar tail (rare) */
            int kb = k / 4;
            uint8_t b0 = r0[kb], b1 = r1[kb], b2 = r2[kb], b3 = r3[kb];
            int8_t x0 = Xq[k + 0];
            d0 += (long)((b0 >> 6) & 0x3) * x0;
            d1 += (long)((b1 >> 6) & 0x3) * x0;
            d2 += (long)((b2 >> 6) & 0x3) * x0;
            d3 += (long)((b3 >> 6) & 0x3) * x0;
            if (k + 1 < K) { int8_t x1 = Xq[k + 1];
                d0 += (long)((b0 >> 4) & 0x3) * x1;
                d1 += (long)((b1 >> 4) & 0x3) * x1;
                d2 += (long)((b2 >> 4) & 0x3) * x1;
                d3 += (long)((b3 >> 4) & 0x3) * x1; }
            if (k + 2 < K) { int8_t x2 = Xq[k + 2];
                d0 += (long)((b0 >> 2) & 0x3) * x2;
                d1 += (long)((b1 >> 2) & 0x3) * x2;
                d2 += (long)((b2 >> 2) & 0x3) * x2;
                d3 += (long)((b3 >> 2) & 0x3) * x2; }
            if (k + 3 < K) { int8_t x3 = Xq[k + 3];
                d0 += (long)((b0 >> 0) & 0x3) * x3;
                d1 += (long)((b1 >> 0) & 0x3) * x3;
                d2 += (long)((b2 >> 0) & 0x3) * x3;
                d3 += (long)((b3 >> 0) & 0x3) * x3; }
        }
        Y[m + 0] = (float)(d0 - sx_sum) * out_scale;
        Y[m + 1] = (float)(d1 - sx_sum) * out_scale;
        Y[m + 2] = (float)(d2 - sx_sum) * out_scale;
        Y[m + 3] = (float)(d3 - sx_sum) * out_scale;
    }

    for (int m = M4; m < M; m++) {                   /* <4 leftover rows */
        long dot = gk_ternary_row_dot(W_packed + (size_t)m * rowb,
                                      Xq, K, K32, ones16);
        Y[m] = (float)(dot - sx_sum) * out_scale;
    }
}
#endif  /* GK_HAVE_AVX2 */

static void
gk_matvec_scalar(int M, int K, const uint8_t *W_packed, float w_scale,
                 const float *X, float *Y) {
    int8_t *Xq = gk_xq_scratch(K);
    if (Xq == NULL) {              /* scratch alloc failed: defined zero output */
        for (int m = 0; m < M; m++) Y[m] = 0.0f;
        return;
    }
    long sx_sum;
    float x_scale = gk_quantize_x(K, X, Xq, &sx_sum);
    float out_scale = x_scale * w_scale;
    if (x_scale == 0.0f) {
        for (int m = 0; m < M; m++) Y[m] = 0.0f;
        return;
    }
    for (int m = 0; m < M; m++) {
        const uint8_t *row = W_packed + m * (K / 4);
        long dot = 0;
        for (int k = 0; k < K; k += 4) {
            uint8_t b = row[k / 4];
            dot += (long)((b >> 6) & 0x3) * (long)Xq[k + 0];
            dot += (long)((b >> 4) & 0x3) * (long)Xq[k + 1];
            dot += (long)((b >> 2) & 0x3) * (long)Xq[k + 2];
            dot += (long)((b >> 0) & 0x3) * (long)Xq[k + 3];
        }
        Y[m] = (float)(dot - sx_sum) * out_scale;
    }
}

/* Public: fp32-in / fp32-out. Y = w_scale * (unpacked_W @ X), with X
 * round-tripped through int8 absmax (the regime BitNet was trained in). */
void
gk_matvec(int M, int K, const uint8_t *W_packed, float w_scale,
          const float *X, float *Y) {
#if GK_HAVE_AVX2
    gk_matvec_avx2(M, K, W_packed, w_scale, X, Y);
#else
    gk_matvec_scalar(M, K, W_packed, w_scale, X, Y);
#endif
}

/* Scalar reference, exposed for correctness testing. */
void
gk_matvec_scalar_pub(int M, int K, const uint8_t *W_packed, float w_scale,
                     const float *X, float *Y) {
    gk_matvec_scalar(M, K, W_packed, w_scale, X, Y);
}

/*
 * int8 x int8 LM-head matvec. The tied LM head (token_embd, 1.3 GB fp32) is
 * row-quantized to int8 with a per-row scale; streaming it at int8 reads 4x
 * fewer bytes than fp32 on the single largest tensor. Y[n] = row_scale[n] *
 * x_scale * sum_k W[n,k]*Xq[k], where Xq is X round-tripped through int8.
 */
#if GK_HAVE_AVX2
static inline long
gk_dot_i8(const int8_t *w, const int8_t *x, int K) {
    int K32 = K & ~31;
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < K32; k += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + k));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + k));
        __m256i wlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv));
        __m256i whi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1));
        __m256i xlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
        __m256i xhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wlo, xlo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(whi, xhi));
    }
    long dot = gk_reduce_add_epi32(acc);
    for (int k = K32; k < K; k++) dot += (long)w[k] * (long)x[k];
    return dot;
}
#endif

void
gk_head_matvec(int N, int K, const int8_t *W, const float *row_scale,
               const float *X, float *Y) {
    int8_t *Xq = gk_xq_scratch(K);
    if (Xq == NULL) {              /* scratch alloc failed: defined zero output */
        for (int n = 0; n < N; n++) Y[n] = 0.0f;
        return;
    }
    long sx_sum;
    float x_scale = gk_quantize_x(K, X, Xq, &sx_sum);
    (void)sx_sum;  /* full-range int8 head: no +128 code-bias trick (would
                    * overflow maddubs int16); gk_dot_i8 widens to int16. */
    if (x_scale == 0.0f) {
        for (int n = 0; n < N; n++) Y[n] = 0.0f;
        return;
    }
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        if((long)N * K >= GK_OMP_THRESHOLD)
    #endif
    for (int n = 0; n < N; n++) {
        const int8_t *row = W + (size_t)n * K;
#if GK_HAVE_AVX2
        long dot = gk_dot_i8(row, Xq, K);
#else
        long dot = 0;
        for (int k = 0; k < K; k++) dot += (long)row[k] * (long)Xq[k];
#endif
        Y[n] = (float)dot * row_scale[n] * x_scale;
    }
}
