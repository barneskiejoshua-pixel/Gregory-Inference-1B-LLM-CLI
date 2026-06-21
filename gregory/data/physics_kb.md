# Gregory physics knowledge base -- PHY 1 (mechanics) and PHY 2
# (thermodynamics, electricity & magnetism, fluids, waves).
#
# Same entry format as automotive_kb.md:
#   ## <topic> | tags: <comma,separated,keywords>
#   <body, with key equations and engineering/mechanic relevance>

## Units and Dimensional Analysis | tags: physics, units, SI, dimensions, conversion
Physics uses SI base units: meter (m), kilogram (kg), second (s), ampere (A),
kelvin (K). Derived units include newton (N = kg*m/s^2), joule (J = N*m), watt
(W = J/s), pascal (Pa = N/m^2). Dimensional analysis checks equations: both
sides must share units. Mechanics tip: torque specs may be given in N*m or
lb-ft (1 lb-ft = 1.356 N*m); tire pressure in psi, kPa, or bar (1 bar = 100 kPa
= 14.5 psi).

## Kinematics | tags: physics, kinematics, velocity, acceleration, motion, PHY1
Kinematics describes motion without forces. For constant acceleration a:
v = v0 + a*t; x = x0 + v0*t + (1/2)*a*t^2; v^2 = v0^2 + 2*a*(x - x0). Velocity
is the rate of change of position; acceleration the rate of change of velocity.
Engineering use: 0-100 km/h times, braking distance, and acceleration loads all
come from these relations.

## Projectile and 2D Motion | tags: physics, projectile, vectors, kinematics, PHY1
Two-dimensional motion separates into independent x and y components. With
gravity only, horizontal velocity is constant and vertical motion has
a = -g (9.81 m/s^2). Vectors add component-wise; magnitude is sqrt(x^2 + y^2).
Useful for trajectory, suspension travel geometry, and any problem combining
horizontal and vertical motion.

## Newton's Laws of Motion | tags: physics, newton, force, dynamics, PHY1
First law: an object keeps its velocity unless a net force acts (inertia).
Second law: F_net = m*a (net force equals mass times acceleration). Third law:
forces come in equal, opposite pairs. These underlie every load and reaction in
a vehicle -- traction, braking, cornering, and the reaction forces at mounts and
fasteners.

## Friction | tags: physics, friction, traction, normal force, PHY1
Friction opposes relative motion. Static friction up to f_s <= mu_s*N; kinetic
friction f_k = mu_k*N, where N is the normal force and mu the coefficient.
Maximum tire grip and brake clamping both depend on the friction coefficient and
the normal (clamp or contact) force. This is why weight transfer and downforce
change available grip.

## Work, Energy, and Power | tags: physics, work, energy, power, PHY1
Work W = F*d*cos(theta) (force through a distance). Kinetic energy
KE = (1/2)*m*v^2; gravitational potential energy PE = m*g*h. The work-energy
theorem: net work equals the change in kinetic energy. Power P = W/t = F*v is
the rate of doing work (watts; 1 hp = 746 W). Braking must dissipate the car's
KE as heat -- doubling speed quadruples the energy to absorb.

## Conservation of Energy | tags: physics, energy conservation, efficiency, PHY1
Energy is neither created nor destroyed; it transforms. In a closed system
mechanical energy (KE + PE) is conserved when only conservative forces act;
friction converts mechanical energy to heat. Efficiency = useful output / total
input. Regenerative braking recovers part of the kinetic energy that friction
brakes would otherwise waste as heat.

## Momentum and Impulse | tags: physics, momentum, impulse, collision, PHY1
Momentum p = m*v is conserved in collisions with no external force. Impulse
J = F*t = change in momentum. Spreading a collision over more time (crumple
zones, airbags) lowers the peak force for the same momentum change -- the core
physics of crashworthiness. Elastic collisions conserve kinetic energy;
inelastic ones do not.

## Circular Motion | tags: physics, circular motion, centripetal, cornering, PHY1
An object moving in a circle of radius r at speed v has centripetal
acceleration a = v^2/r directed inward, requiring force F = m*v^2/r. In
cornering, tire grip supplies this force; exceed it and the car slides. Maximum
cornering speed scales with sqrt(mu*g*r), which is why grip and corner radius set
the limit.

## Rotational Dynamics and Torque | tags: physics, torque, rotation, inertia, PHY1
Torque tau = r*F*sin(theta) is the rotational analog of force; the rotational
Newton's law is tau = I*alpha (moment of inertia times angular acceleration).
Engine output is torque (N*m) and power = torque * angular speed (P = tau*omega).
Fastener torque specs set clamping force via bolt geometry; a torque wrench
applies a controlled tau.

## Angular Momentum | tags: physics, angular momentum, gyroscope, rotation, PHY1
Angular momentum L = I*omega is conserved without external torque. Spinning
wheels and crankshafts store rotational energy (1/2)*I*omega^2 and resist
changes in orientation (gyroscopic effect). Flywheels use this to smooth engine
power delivery and store energy.

## Gravitation and Weight | tags: physics, gravity, weight, mass, PHY1
Weight is the gravitational force W = m*g, with g = 9.81 m/s^2 near Earth's
surface; mass is the amount of matter and is constant. Center of gravity
location governs weight distribution, load transfer under braking and
cornering, and rollover tendency -- a lower CG (e.g. an EV skateboard battery)
improves stability.

## Statics and Equilibrium | tags: physics, statics, equilibrium, free body, PHY1
A body in equilibrium has zero net force and zero net torque: sum(F) = 0 and
sum(tau) = 0. Free-body diagrams resolve the forces. Statics sizes structures,
finds reaction loads at mounts and bearings, and determines how weight splits
between axles -- the basis of chassis and suspension load analysis.

## Simple Harmonic Motion | tags: physics, SHM, oscillation, resonance, vibration, PHY1
A mass on a spring oscillates with angular frequency omega = sqrt(k/m) and
period T = 2*pi*sqrt(m/k). Damping removes energy; resonance occurs when forcing
matches the natural frequency, amplifying motion. Suspension is a damped
spring-mass system; NVH engineering avoids resonances in the body, driveline,
and mounts.

## Stress, Strain, and Elasticity | tags: physics, stress, strain, modulus, materials, PHY1
Stress = force / area (Pa); strain = fractional deformation. In the elastic
region stress = E * strain (Hooke's law), where E is Young's modulus. Beyond the
yield point material deforms permanently; the ultimate strength precedes
fracture. These set fastener preload, shaft sizing, and the deformation of crash
structures.

## Fluid Statics and Pressure | tags: physics, fluid, pressure, pascal, hydraulics, PHY2
Pressure P = F/A (Pa). In a fluid, pressure increases with depth: P = rho*g*h.
Pascal's principle: pressure applied to a confined fluid transmits undiminished,
so a small force on a small piston yields a large force on a large piston
(F1/A1 = F2/A2). This is how hydraulic brakes, jacks, and clutches multiply
force.

## Buoyancy | tags: physics, buoyancy, archimedes, density, PHY2
Archimedes' principle: a submerged or floating body is buoyed up by a force
equal to the weight of displaced fluid, F = rho_fluid * g * V_displaced. It
explains floats in carburetors and fuel-level senders, and hydrometer testing of
coolant and battery electrolyte specific gravity.

## Fluid Dynamics: Continuity and Bernoulli | tags: physics, bernoulli, flow, continuity, PHY2
Continuity: for incompressible flow A1*v1 = A2*v2 (narrower passage, faster
flow). Bernoulli: P + (1/2)*rho*v^2 + rho*g*h = constant along a streamline, so
faster flow means lower pressure. This underlies venturis, carburetors,
airflow over wings (lift/downforce), and intake/exhaust gas flow.

## Viscosity and Flow Resistance | tags: physics, viscosity, laminar, turbulent, oil, PHY2
Viscosity measures a fluid's resistance to shear; it falls as temperature rises.
Flow is laminar at low Reynolds number and turbulent at high. Oil viscosity
grades (e.g. 5W-30) describe cold and hot behavior; pressure drop in lines and
coolers depends on viscosity, flow rate, and passage size.

## Temperature and Heat | tags: physics, temperature, heat, specific heat, thermal, PHY2
Temperature measures average molecular kinetic energy; heat Q is energy
transferred due to a temperature difference. Q = m*c*dT, where c is specific
heat. Conversions: K = C + 273.15; F = C*9/5 + 32. Cooling-system sizing,
brake-fade thresholds, and warm-up behavior all follow from heat capacity and
flow.

## Thermal Expansion | tags: physics, thermal expansion, clearance, tolerance, PHY2
Most materials expand when heated: dL = alpha * L * dT (linear), with alpha the
expansion coefficient. Engineers leave clearances for it -- piston-to-bore
clearance, valve lash, brake-disc growth, and press/shrink fits all account for
thermal expansion. Aluminum expands about twice as much as steel.

## Laws of Thermodynamics | tags: physics, thermodynamics, first law, entropy, PHY2
First law: energy is conserved; dU = Q - W (internal energy change equals heat
in minus work out). Second law: heat flows from hot to cold and no engine is
100% efficient; entropy of an isolated system increases. These bound how much of
a fuel's energy an engine can turn into work.

## Heat Engines and Efficiency | tags: physics, heat engine, carnot, efficiency, otto, PHY2
A heat engine converts heat to work between hot and cold reservoirs. Thermal
efficiency = W_out / Q_in. The Carnot limit eta = 1 - T_cold/T_hot (kelvin) caps
any engine. The Otto cycle (gasoline) efficiency rises with compression ratio;
real engines lose to friction, pumping, and heat rejection, so typical brake
thermal efficiency is roughly 25-40%.

## Heat Transfer | tags: physics, heat transfer, conduction, convection, radiation, PHY2
Heat moves three ways: conduction through solids (Fourier's law, rate ~ k*A*dT/
L), convection by fluid motion (rate ~ h*A*dT), and radiation (~ emissivity *
sigma * A * T^4). Radiators use forced convection and high surface area; heat
shields block radiation; thermal paste improves conduction at contacts.

## Electric Charge and Coulomb's Law | tags: physics, charge, coulomb, electrostatics, PHY2
Like charges repel, opposite attract. Coulomb's law: F = k*q1*q2/r^2. Charge is
quantized and conserved. Static discharge (ESD) can destroy automotive
electronic modules, which is why technicians ground themselves before handling
ECUs and sensors.

## Electric Field and Potential | tags: physics, electric field, voltage, potential, PHY2
An electric field E exerts force F = q*E on charge. Electric potential
(voltage) is energy per unit charge; potential difference drives current.
V = energy / charge (volts = J/C). A 12 V battery supplies 12 J per coulomb;
EV traction packs run at 400-800 V, which is why their service demands special
high-voltage precautions.

## Capacitance | tags: physics, capacitor, capacitance, farad, PHY2
A capacitor stores charge: Q = C*V, energy E = (1/2)*C*V^2. Capacitors smooth
voltage ripple, filter noise, and buffer fast current demands in electronics and
inverters. EV/inverter DC-link capacitors hold dangerous charge after shutdown,
so a bleed-down wait is required before service.

## Current, Resistance, and Ohm's Law | tags: physics, current, resistance, ohm, voltage, PHY2
Current I is charge flow per second (amperes). Ohm's law: V = I*R. Resistance
rises with length and falls with cross-section (R = rho*L/A) and increases with
temperature in metals. Undersized or corroded wiring adds resistance and voltage
drop -- a frequent cause of dim lights, slow cranking, and false sensor readings.

## DC Circuits and Electrical Power | tags: physics, circuit, series, parallel, power, PHY2
In series, resistances add and current is shared; in parallel, voltage is shared
and conductances add. Kirchhoff's laws: currents into a node sum to zero, and
voltages around a loop sum to zero. Power P = V*I = I^2*R = V^2/R (watts). Fuse
and wire sizing follow from expected current and the I^2*R heating it causes.

## Magnetism and Magnetic Force | tags: physics, magnetism, magnetic field, motor, PHY2
A magnetic field B exerts a force on moving charge (F = q*v*B) and on a
current-carrying wire (F = B*I*L). This force is what turns electric motors and
moves solenoid plungers and fuel injectors. Field strength and current set the
torque an electric machine can produce.

## Electromagnetic Induction | tags: physics, induction, faraday, lenz, alternator, PHY2
Faraday's law: a changing magnetic flux induces a voltage, EMF = -d(flux)/dt;
Lenz's law sets its opposing direction. Induction generates electricity in
alternators and EV motor regeneration, and underlies ignition coils,
transformers, and inductive (crank/cam) speed sensors.

## Inductance and AC Basics | tags: physics, inductance, AC, frequency, inverter, PHY2
An inductor opposes changes in current; V = L*dI/dt, energy = (1/2)*L*I^2.
Alternating current reverses periodically (frequency in Hz). Inverters synthesize
AC to drive traction motors; ignition coils use inductance to make a high-voltage
spark. Reactance makes impedance frequency-dependent in AC circuits.

## Waves and Sound | tags: physics, waves, frequency, sound, NVH, resonance, PHY2
A wave carries energy; speed = frequency * wavelength (v = f*lambda). Sound is a
pressure wave; loudness is measured in decibels (a log scale). Resonance
amplifies specific frequencies. NVH diagnosis identifies noises by frequency
(e.g. a wheel-speed-related drone vs an engine-order vibration) to find the
source.

## Geometric Optics | tags: physics, optics, reflection, refraction, lens, sensor, PHY2
Light reflects (angle in = angle out) and refracts when changing media (Snell's
law, n1*sin1 = n2*sin2). Lenses and mirrors form images used in headlamps,
cameras, and lidar. Refraction and focal length govern how ADAS cameras and
optical sensors image the road.
