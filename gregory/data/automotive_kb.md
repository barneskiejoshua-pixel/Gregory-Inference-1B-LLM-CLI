# Gregory automotive-engineering knowledge base
#
# One entry per block. Format:
#   ## <topic> | tags: <comma,separated,keywords>
#   <body text (any number of lines until the next "## ")>
#
# Entries are concise, factual grounding snippets retrieved and injected into
# the prompt at answer time. Keep them accurate and source-neutral.

## Automotive Design | tags: design, aerodynamics, styling, ergonomics, packaging
Automotive design balances form, function, aerodynamics, ergonomics, and
manufacturability. Key metrics include the drag coefficient (Cd), frontal area,
interior packaging efficiency, and weight distribution. Styling (exterior and
interior) is constrained by crash structures, pedestrian-safety rules, and
aerodynamic targets that affect efficiency and high-speed stability.

## Vehicle Dynamics | tags: dynamics, handling, understeer, oversteer, yaw, slip
Vehicle dynamics studies how a vehicle responds to driver and road inputs in
the longitudinal, lateral, and vertical directions. Understeer is when the
front tires lose grip first and the car runs wide; oversteer is when the rear
loses grip and the tail slides. Behaviour depends on weight transfer, tire slip
angles, the yaw moment, and the center of gravity height.

## Powertrain | tags: powertrain, engine, transmission, torque, driveline, ratio
The powertrain converts stored energy into wheel torque: engine or motor,
transmission, driveshafts, differential, and axles. Overall gear ratio trades
acceleration against top speed. Tractive effort at the wheels equals engine
torque times the total ratio times driveline efficiency, divided by the rolling
radius.

## Internal Combustion Engine | tags: engine, ICE, four-stroke, otto, diesel, compression
A four-stroke engine cycles intake, compression, power, and exhaust. Spark-
ignition (Otto cycle) engines burn premixed gasoline; compression-ignition
(Diesel cycle) engines inject fuel into hot compressed air. Higher compression
ratio raises thermal efficiency but risks knock in gasoline engines, which is
limited by fuel octane rating.

## Combustion and Fuel | tags: combustion, fuel, AFR, lambda, stoichiometric, octane, cetane
The stoichiometric air-fuel ratio for gasoline is about 14.7:1 by mass; lambda
(λ) is the actual AFR divided by stoichiometric, so λ=1 is stoichiometric, λ>1
is lean, λ<1 is rich. Octane rating measures gasoline knock resistance; cetane
rating measures diesel ignition quality. Closed-loop fueling uses an oxygen
(lambda) sensor to hold λ near 1 for the catalytic converter.

## Transmission | tags: transmission, manual, automatic, CVT, DCT, gearbox, clutch
Transmissions multiply engine torque and match engine speed to road speed.
Types include manual (driver clutch), torque-converter automatic, dual-clutch
(DCT) for fast shifts, and continuously variable (CVT) for a seamless ratio.
EVs commonly use a single fixed reduction gear because electric motors deliver
torque across a wide speed range.

## Suspension | tags: suspension, macpherson, wishbone, damper, spring, roll, ride
Suspension controls wheel motion for ride comfort and tire contact. Common
layouts are MacPherson strut and double wishbone. Spring rate sets ride
stiffness; the damper (shock) controls oscillation; anti-roll bars limit body
roll in cornering. Tuning trades ride comfort against handling and body
control.

## Steering | tags: steering, rack, ackermann, EPS, caster, camber, toe, alignment
Most cars use rack-and-pinion steering, increasingly with electric power assist
(EPS). Ackermann geometry turns the inner wheel more than the outer in a turn.
Wheel alignment angles: camber (lean from vertical), caster (steering-axis
tilt, gives straight-line stability), and toe (in/out), each affecting tire
wear and handling.

## Braking | tags: braking, brake, disc, drum, ABS, bias, regenerative, hydraulic
Friction brakes convert kinetic energy to heat via discs or drums actuated by
hydraulic pressure. Anti-lock braking (ABS) modulates pressure to prevent wheel
lockup and keep steering control. Brake bias distributes force front-to-rear.
Electrified vehicles add regenerative braking, recovering energy through the
motor before the friction brakes engage (blended braking).

## Tires and Grip | tags: tire, tyre, grip, slip ratio, friction circle, contact patch
Tires transmit all driving, braking, and cornering forces through small contact
patches. Longitudinal slip ratio and lateral slip angle generate force up to
the friction limit. The friction circle shows that combined braking and
cornering share one total grip budget. Grip depends on load, temperature,
compound, and road surface.

## Hybrid and Electric Vehicles | tags: EV, hybrid, BEV, HEV, PHEV, motor, inverter
A BEV runs purely on a battery and electric motor; an HEV combines an engine
with a motor and cannot plug in; a PHEV adds a larger, grid-chargeable battery.
The inverter converts DC battery power to AC for the traction motor and controls
torque. Electric drive gives instant torque and high efficiency but depends on
battery energy density and charging.

## Battery Systems | tags: battery, li-ion, NMC, LFP, BMS, SOC, kWh, thermal runaway
Most EVs use lithium-ion cells; common chemistries are NMC (high energy) and
LFP (lower cost, long life, safer). Capacity is rated in kWh; state of charge
(SOC) is the remaining fraction. The battery management system (BMS) balances
cells, enforces voltage/current/temperature limits, and prevents thermal
runaway. Pack thermal management (liquid cooling) protects life and fast-charge
capability.

## EV Charging | tags: charging, AC, DC fast, CCS, CHAdeMO, NACS, J1772, kW
AC charging uses the onboard charger (typically 7-22 kW); DC fast charging
bypasses it to feed the pack directly (50-350+ kW). Connector standards include
J1772 (AC, North America), CCS (DC), CHAdeMO (legacy DC), and NACS/Tesla.
Charge rate tapers as SOC rises to protect the cells, so 10-80% is the typical
fast-charge window.

## Alternative Fuels | tags: hydrogen, fuel cell, biofuel, CNG, e-fuel, FCEV
Beyond gasoline and diesel: hydrogen fuel-cell vehicles (FCEV) generate
electricity from H2 and emit only water; biofuels (ethanol, biodiesel) blend
with or replace fossil fuel; compressed natural gas (CNG) and synthetic e-fuels
offer lower-carbon options. Each trades energy density, infrastructure, cost,
and well-to-wheel emissions.

## ADAS and Autonomy | tags: ADAS, autonomy, SAE J3016, levels, lidar, radar, sensor fusion
SAE J3016 defines driving-automation levels 0-5: 0 none, 1 driver assistance,
2 partial (hands-on, e.g. adaptive cruise + lane centering), 3 conditional, 4
high (no driver in its domain), 5 full. ADAS fuses cameras, radar, lidar, and
ultrasonic sensors. Functions include AEB (automatic emergency braking), lane
keeping, and adaptive cruise control.

## Automotive Electronics | tags: electronics, ECU, sensor, actuator, microcontroller
Modern vehicles run dozens of electronic control units (ECUs): networked
microcontrollers reading sensors and driving actuators. Examples include the
engine/powertrain control module, body control module, ABS/ESC controller, and
infotainment. Domain and zonal architectures consolidate functions to cut
wiring and enable software updates.

## In-Vehicle Networks: CAN | tags: CAN, CAN bus, network, arbitration, bitrate
Controller Area Network (CAN) is the dominant in-vehicle bus: a two-wire
differential, multi-master, message-based protocol. Identifiers are 11-bit
(standard) or 29-bit (extended); lower IDs win bus arbitration (priority).
Classic CAN runs up to 1 Mbit/s; CAN FD adds a faster data phase and larger
payloads. Messages are broadcast, not addressed to a node.

## In-Vehicle Networks: LIN, FlexRay, Ethernet | tags: LIN, FlexRay, automotive ethernet, network
LIN is a low-cost single-wire bus for simple body functions (mirrors, windows).
FlexRay is a deterministic, time-triggered bus (up to 10 Mbit/s) used for
chassis/safety. Automotive Ethernet (100BASE-T1, 1000BASE-T1, multi-gig) carries
high-bandwidth ADAS and infotainment data and underpins zonal architectures and
service-oriented (SOME/IP) communication.

## OBD and Diagnostics | tags: OBD, OBD-II, DTC, PID, UDS, diagnostics, J1979
On-board diagnostics (OBD-II) standardizes a 16-pin connector and emissions
monitoring. Diagnostic trouble codes (DTCs) identify faults; PIDs query live
data (SAE J1979). Unified Diagnostic Services (UDS, ISO 14229) supports deeper
service, ECU flashing, and security access. Scan tools read codes and freeze-
frame data to guide repair.

## Functional Safety: ISO 26262 | tags: functional safety, ISO 26262, ASIL, HARA, FMEA, safety goal
ISO 26262 is the road-vehicle functional-safety standard for electrical and
electronic systems, adapting IEC 61508. A hazard analysis and risk assessment
(HARA) assigns each hazard an ASIL (Automotive Safety Integrity Level) from A
(lowest) to D (highest), based on severity, exposure, and controllability; QM
means no ASIL required. Safety goals flow into requirements verified through the
safety lifecycle, supported by FMEA and FTA.

## AUTOSAR | tags: AUTOSAR, classic platform, adaptive platform, RTE, BSW, SWC
AUTOSAR is a standardized automotive software architecture. The Classic Platform
targets deeply embedded, real-time ECUs: application software components (SWCs)
communicate through the Runtime Environment (RTE) over standardized Basic
Software (BSW) and a microcontroller abstraction layer. The Adaptive Platform
targets high-performance, service-oriented computing (POSIX, SOME/IP) for ADAS
and connectivity.

## Software and Systems Engineering | tags: software, systems engineering, V-model, MISRA, requirements
Automotive software follows the V-model: requirements and design descend the
left arm, integration and verification climb the right, with traceability
throughout. Coding standards such as MISRA C constrain risky language features
for safety. Process frameworks (ASPICE) assess capability. Increasingly,
software-defined vehicles deliver features and fixes via over-the-air updates.

## Materials Science | tags: materials, steel, aluminum, magnesium, CFRP, composites, lightweighting
Material choice trades strength, weight, cost, and formability. High-strength
steels dominate body structures; aluminum and magnesium cut mass; carbon-fiber-
reinforced polymer (CFRP) gives the best stiffness-to-weight at high cost.
Lightweighting improves efficiency, acceleration, and range, and is central to
EV design where battery mass is large.

## Manufacturing Processes | tags: manufacturing, stamping, casting, forging, welding, assembly
Vehicle manufacturing spans stamping sheet metal, casting and forging
components, machining, and welding (spot welding for body-in-white). Final
assembly runs on a moving line with just-in-time logistics. Giga-casting
replaces many stamped parts with large single aluminum castings. Quality
control uses statistical process control and end-of-line testing.

## NVH | tags: NVH, noise, vibration, harshness, modal, damping, refinement
NVH engineering controls noise, vibration, and harshness for refinement and
durability. Sources include the powertrain, road, wind, and driveline. Tools
include modal analysis (natural frequencies and mode shapes), damping
treatments, isolation mounts, and absorption. EVs shift NVH focus to motor
whine, gear noise, and previously masked wind and road noise.

## Thermodynamics and Cooling | tags: thermodynamics, heat transfer, cooling, radiator, thermal management
Engines and electric drives reject heat that must be managed. Cooling systems
use coolant loops, radiators, fans, and oil coolers. Thermal management governs
efficiency, durability, emissions, and (in EVs) battery life and fast-charge
capability. Heat pumps improve EV cabin heating efficiency versus resistive
heaters.

## Crashworthiness and Safety | tags: crash, crashworthiness, crumple zone, airbag, NCAP, restraint
Crashworthiness protects occupants by managing crash energy: crumple zones
deform to absorb energy while a rigid safety cage preserves survival space.
Restraint systems (seatbelts with pretensioners and load limiters, airbags)
manage occupant deceleration. Programs like Euro NCAP and US NCAP rate
protection through standardized crash tests.

## Emissions and Aftertreatment | tags: emissions, catalytic converter, EGR, DPF, SCR, NOx, aftertreatment
Exhaust aftertreatment reduces pollutants. A three-way catalytic converter (at
λ=1) cuts CO, HC, and NOx for gasoline engines. Diesels use exhaust-gas
recirculation (EGR), a diesel particulate filter (DPF), and selective catalytic
reduction (SCR) with urea (AdBlue) to control NOx and soot. Gasoline direct-
injection engines may add a gasoline particulate filter (GPF).

## Emissions Regulation | tags: emissions regulation, Euro 6, WLTP, EPA, CO2, fleet
Regulations cap tailpipe pollutants and CO2. Europe uses Euro standards (Euro
6/7) and the WLTP test cycle; the US uses EPA and CARB standards with EPA test
cycles. Fleet-average CO2 / fuel-economy targets (CO2 g/km, CAFE mpg) push
electrification. Homologation certifies a vehicle meets the applicable market
regulations before sale.

## Aerodynamics | tags: aerodynamics, drag, lift, downforce, Cd, CFD, wind tunnel
Aerodynamics governs efficiency, stability, and cooling airflow. Drag force
scales with the drag coefficient (Cd), frontal area, and the square of speed,
so it dominates highway energy use. Lift and downforce affect high-speed
stability; performance cars use wings and diffusers for downforce. Development
uses CFD simulation and wind-tunnel testing.

## Vehicle Architecture | tags: architecture, platform, body-on-frame, unibody, skateboard
A platform shares structure and components across models. Body-on-frame mounts
the body on a separate ladder frame (trucks, heavy SUVs); unibody (monocoque)
integrates body and structure for lighter, stiffer passenger cars. EVs often use
a flat skateboard platform with the battery in the floor, freeing interior space
and lowering the center of gravity.

## Chassis and Structure | tags: chassis, frame, subframe, torsional rigidity, stiffness
The chassis carries loads and locates the suspension. Torsional rigidity (body
stiffness) underpins handling precision, NVH, and durability. Subframes mount
suspension and powertrain and isolate vibration. The structure must also provide
crash load paths and, in EVs, protect and integrate the battery pack.

## Control Engineering | tags: control, PID, ESC, traction control, stability, feedback
Control systems regulate vehicle behaviour using feedback. PID controllers are
common for engine idle, cruise, and thermal loops. Electronic stability control
(ESC) compares intended and actual yaw and brakes individual wheels to prevent
skids; traction control limits wheel spin. Model-based control and state
estimation are increasingly used for electrified and automated driving.

## Testing and Validation | tags: testing, validation, dyno, HIL, durability, homologation
Validation spans component to vehicle level: engine/chassis dynamometers,
hardware-in-the-loop (HIL) for ECUs, durability and fatigue rigs, climatic and
proving-ground tests, and crash testing. Simulation (MIL/SIL/HIL) reduces
physical prototypes. Homologation confirms regulatory compliance for each
market before production.

## Connected Vehicles and V2X | tags: connected, V2X, V2V, V2I, telematics, OTA
Connected-vehicle technology links cars to each other (V2V), infrastructure
(V2I), and the cloud (collectively V2X). Telematics enables remote diagnostics,
fleet management, and emergency call. Over-the-air (OTA) updates patch software
and add features. V2X can warn of hazards beyond line of sight, supporting
safety and traffic efficiency.

## Standards and Organizations | tags: standards, SAE, ISO, UNECE, NHTSA, NCAP, organizations
Key bodies shape automotive engineering: SAE International publishes technical
standards (e.g. J1939, J1772, J3016) and the AUTOSAR/ISO ecosystems; ISO issues
international standards (26262 safety, 14229 UDS, 11898 CAN); UNECE harmonizes
vehicle regulations; NHTSA regulates US vehicle safety; Euro NCAP and US NCAP
rate crash safety.
