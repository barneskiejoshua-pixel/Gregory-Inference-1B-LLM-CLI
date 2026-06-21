# Gregory developer entry points. `make ci` is THE gate.
PY ?= python3

.PHONY: test nparc nparc-baseline ci clean kb ingest

test:            ## Fast deterministic gate (model-free)
	@$(PY) -m pytest -m "not slow"

kb:              ## List the automotive knowledge-base topics
	@$(PY) -c "import sys;sys.path.insert(0,'.');from gregory.cli import main;main()" kb

ingest:          ## Ingest docs into the KB: make ingest IN=path OUT=name
	@$(PY) scripts/ingest_docs.py $(IN) --out $(OUT)

nparc:           ## NPARC code-standard compliance gate
	@$(PY) scripts/nparc_compliance.py --strict

nparc-baseline:  ## (Re)write the NPARC ratchet baseline
	@$(PY) scripts/nparc_compliance.py --set-baseline

ci: nparc test   ## What CI runs: standards gate, then tests

clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null \
		|| true
	@rm -rf .pytest_cache
