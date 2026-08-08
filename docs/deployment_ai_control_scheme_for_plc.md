# AI Model Control Scheme for PLC / Microgrid Integration

## Quick Integration Summary

This is the short file I/O spec for connecting the PC-side AI model to the PLC /
microgrid controller. The PC AI model only reads sensor data and writes a high
level command; PLC-side safety override remains required.

### 1. Input: `Data.txt`

PLC / vendor controller writes the latest sensor data to `Data.txt`. Preferred
format:

```text
YYYYMMDDHHmmSS
SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
LoadV,LoadI,LoadP,GridV,GridI,GridP,
ID,SOC,BV,ChargeV,ChargeI,Temp,Speed,
```

Please provide these data if available: timestamp, battery physical ID, battery
voltage/current/power, SoC, load power, PV/solar power, grid power, and mode /
status / flow feedback. If battery power is not reported directly, the PC
estimates it from battery voltage and current.

Current units: voltage = `0.01 V`; current = `mA`; power = `mW`; SoC = `0.1%`;
temperature = `0.1 deg C`; speed/flow feedback = `0.1%`.

### 2. Output: `Command.txt`

PC AI writes `Command.txt`:

```text
{situation_code}
YYYYMMDDHHmmSS,{load_count}
PP,power_mW,flow_percent,
```

`PP` is the physical battery ID, normally `01`. `power_mW` is a non-negative
power magnitude in `mW`. Charge/discharge direction is determined by
`situation_code`, not by a negative power sign.

- `mode 1`: battery discharge, only when allowed by AI and deployment guards.
- `mode 3`: battery charge, normal standby/rest, and pre-measure.
- `mode 4`: shutdown / motor off only. It is not normal standby.
- `mode 2`: not used by the current deployment logic.

### 3. Decision Loop

1. PLC writes or updates `Data.txt`.
2. AI reads `Data.txt` every decision cycle. Current deployment polls about
   every 10 s and makes model decisions over about a 15 min window.
3. Before decision, AI may write zero-power 50% flow for voltage pre-measure
   (`mode 3 + 01,0,50`) and wait about 25 s.
4. AI reads fresh `Data.txt` again.
5. AI writes the final `Command.txt`.
6. PLC executes the latest valid command and may override for safety.

### 4. Important Command Examples

Normal standby / rest:

```text
3
20260712120000,4
01,0,0,
```

Pre-measure before AI decision:

```text
3
20260712120000,4
01,0,50,
```

Charge example, 5000 mW, 60% flow:

```text
3
20260712121500,4
01,5000,60,
```

Discharge example, 3000 mW, 60% flow:

```text
1
20260712123000,4
01,3000,60,
```

The examples keep the physical battery ID `01` even for zero-power commands,
because zero-power flow commands still need to target the physical flow path.

### 5. Open Questions for Antoine / PLC Side

1. The exact final `Data.txt` schema, field order, units, and whether grid/PV can
   be measured at the same time.
2. Whether `Command.txt` will be the final interface, or whether PLC registers /
   Modbus / another channel will mirror the same fields.
3. ACK / rejection behavior: can the PLC report command accepted, rejected, or
   active mode?
4. Timeout behavior: what should the PLC do if `Command.txt` is stale or the PC
   stops writing commands?
5. Command sign and unit semantics: confirm `power_mW` is non-negative and
   charge/discharge direction comes from the situation code.
6. Whether the PLC supports the pre-measure command `mode 3 + 01,0,50` for about
   25 s before the AI reads fresh data.
7. Exact semantics of `mode 3` and `mode 4`: confirm `mode 3` is normal
   grid/PV/charge/rest/pre-measure path, and `mode 4` is shutdown / motor off
   only.

## Appendix: Existing Background Notes

### Purpose

This document summarizes the current PC-side AI model control scheme used in
the P302 microgrid deployment stack. The goal is to help connect the AI model
control functionality on the PC to the physical microgrid system, including the
PLC, sensors, vendor controller, and command interface.

The target integration plan is:

- By the end of July: establish the AI-model-to-microgrid control connection.
- By the end of August: test, validate, and optimize the integrated control loop.

### System Role

The PC AI model is an upper-level Energy Management System (EMS). It decides
energy-management actions such as battery charge, battery discharge, standby,
and pump / flow command levels.

The PC AI model is not a low-level power-electronics controller. It does not
replace inverter, converter, relay, MPPT, or motor driver protection logic. The
PLC / embedded controller should still own the low-level interlocks, actuator
limits, electrical protection, emergency stop, and real-time switching details.

The current control stack is:

```text
Sensors / PLC / vendor controller
        -> Data.txt or equivalent input protocol
        -> PC AI EMS
        -> safety guards and command projection
        -> Command.txt or equivalent output protocol
        -> PLC / vendor controller / actuators
```

The existing implementation is mainly in:

- `control/run_deployment.py`
- `control/io_protocol.py`
- `control/solar_test_collect.py`
- `gui/ai_control_gui.py`

### Timing Sequence

The normal AI decision cycle is approximately 15 minutes.

Current deployment behavior:

1. The vendor controller / PLC writes fresh measurements to `Data.txt` or an
   equivalent input channel.
2. The PC AI process polls measurements about every 10 seconds.
3. The PC aggregates data into a 15-minute decision window.
4. Before each model decision, the PC sends a pre-measure command:
   `mode 3 + physical battery PP + power 0 + flow 50%`.
5. The system waits about 25 seconds for voltage / flow-related measurement
   recovery.
6. The PC reads fresh `Data.txt` data.
7. The AI model computes an EMS action.
8. Deployment safety guards adjust or block unsafe actions.
9. The PC writes the final command to `Command.txt` or an equivalent command
   channel.
10. Between decisions, the deployment loop may refresh the latest command so
    the hardware keeps seeing the intended state.

Normal standby / rest is not the same as pre-measure:

- Standby / rest: `mode 3 + physical PP + power 0 + flow 0%`.
- Pre-measure: `mode 3 + physical PP + power 0 + flow 50%`, held for about
  25 seconds before reading fresh data.

The flow / pump affects voltage measurement. In simulation, the pre-measure
event is mainly logged as metadata; in deployment, the command is actually sent
to the hardware before reading fresh voltage / sensor data.

### Required Sensor Inputs

The AI EMS can operate with `Data.txt` or an equivalent PLC I/O protocol. The
latest parser supports several firmware generations, but the preferred input
fields are:

- Timestamp / time, e.g. `YYYYMMDDHHmmSS`.
- Load voltage, load current, and load power.
- Solar / PV input measurements.
- MPPT output measurements.
- Bus-side PV support measurements when available.
- Grid voltage, grid current, and grid power when available.
- Battery physical ID, e.g. `01`.
- Battery SoC if available from firmware.
- Battery terminal voltage.
- Battery charge voltage if available.
- Battery current / power.
- Battery temperature if available.
- Flow / pump speed / mode / status if available.
- Time-of-use price or a TOU schedule available on the PC side.

Current `Data.txt` formats supported by the code include:

```text
YYYYMMDDHHmmSS
SolarV,SolarI,SolarP,MPPT_V,MPPT_I,MPPT_P,BusV,BusI,BusP,
LoadV,LoadI,LoadP,GridV,GridI,GridP,
ID,SOC,BV,ChargeV,ChargeI,Temp,Speed,
```

Older formats without bus, grid, or charge-voltage fields are also parsed, but
the integration should prefer the complete format above if the PLC can provide
it.

Typical units in the current vendor file are:

- Voltage fields: integer in 0.01 V.
- Current fields: integer in mA.
- Power fields: integer in mW.
- SoC: integer in 0.1%.
- Temperature: integer in 0.1 deg C.
- Flow / speed: integer in 0.1% for sensor feedback, and percent for commands.

### AI Model Outputs

The AI model and deployment layer produce these command concepts:

- Situation / mode code.
- Physical battery ID / PP, e.g. `01`.
- Battery power command.
- Pump / flow percentage.

The current `Command.txt` format is:

```text
{situation_code}
YYYYMMDDHHmmSS,{load_count}
PP,power_mW,flow_percent,
```

Example standby command:

```text
3
20260712120000,4
01,0,0,
```

Example pre-measure command:

```text
3
20260712120000,4
01,0,50,
```

The latest implementation intentionally keeps the physical battery ID even when
power is zero. It should not replace the battery ID with `00`, because a
zero-power command may still need to target the physical pump / flow path during
pre-measure.

### Command Semantics

Current deployment semantics are:

- `mode 1`: battery discharge can serve the load when the command is
  hardware-feasible and passes safety guards.
- `mode 2`: discharge plus grid support is not used by the current deployment
  logic because battery discharge should not be treated as a partial parallel
  assist source.
- `mode 3`: grid / PV path, battery charging, standby / rest, and pre-measure.
- `mode 4`: explicit shutdown / motor off only. It is not normal standby.

Important wording for integration and documentation:

- Do not describe the microgrid as a strict "PV or grid" binary switch.
- PV and grid may co-support the load.
- Battery discharge is different: it should be commanded only when the request
  is hardware-feasible and guarded. The battery should not be treated as a third
  source that partially assists alongside PV / grid.
- Prefer terms such as "PV support increases" and "grid demand decreases"
  rather than "the load is fully supplied by solar", unless exclusive source
  attribution is directly measured.

### Flow / Pump Rules

Current constants in deployment are:

- Normal rest flow: `0%`.
- Pre-measure flow: `50%`.
- Minimum active flow: `60%`.
- Active command range after guarding: clipped to `60-100%` when an active
  battery charge / discharge action is being commanded.

Flow rate is currently part of the command interface and logs. However, on the
small P302 platform, pump / motor energy can be significant relative to battery
power. Therefore, the current hardware deployment should treat flow control
conservatively:

- Use `0%` for rest / standby.
- Use `50%` only for pre-measure voltage recovery before a decision.
- Use at least the active minimum when a real battery action is commanded.
- Keep full real-time flow-rate profit optimization as future work unless the
  motor power budget is validated.

### Safety Guards

The raw model action is not sent directly to hardware. The deployment layer
applies safety guards and logs both the raw model intent and final command.

Current safety guards include:

- Low SoC blocks battery discharge, currently around 20%.
- High SoC blocks charging, currently around 80%.
- High charge voltage blocks charging, currently around 8.8 V.
- Low battery voltage cutoff blocks discharge, currently below 4.2 V.
- Cutoff recovery requires approximately 5.0 V, with cooldown / daily limit
  logic.
- PV-active discharge guard blocks battery discharge while PV support is active.
- Invalid partial-discharge guard blocks a discharge command when the battery
  would only partially cover a feasible load. In that case, the command returns
  to standby / grid-PV path.
- PV-surplus charge guard can require charging to be limited to PV surplus.
- Flow-power guard can limit battery power according to available flow-related
  power semantics from the model configuration.
- Small discharge intent below about 50 mW is treated as standby.
- Firmware / current inconsistency and isolated-load-bus checks are logged and
  can block unsafe discharge paths.
- Optional SoH health lock exists as an integration interface, but should remain
  disabled unless validated.

Load-over-discharge-limit behavior needs careful interpretation:

- Some simulation or older documentation may describe load-over-limit as a
  blocking guard.
- In the latest deployment semantics, measured load above the battery discharge
  limit is a diagnostic warning, not an automatic block. This is because the
  measured load estimate may be conservative when all load banks are on.
- Low voltage cutoff, PV-active discharge, and invalid partial-discharge guards
  can still block discharge even when the load-over-limit condition is only a
  warning.

The PLC / vendor side should also keep independent hard protection for:

- Emergency stop.
- Over-voltage / under-voltage.
- Over-current.
- Battery and motor thermal limits.
- Communication timeout.
- Invalid mode / ID / command-range values.
- Command age / timestamp validation.
- Safe fallback to mode 3 rest or vendor-defined safe state.

### PLC Integration Checklist

Antoine / vendor team should confirm or implement the following:

- Confirm whether the integration will use the current `Data.txt` /
  `Command.txt` files or an equivalent PLC communication channel.
- If an equivalent channel is used, keep the same logical fields and units or
  provide a deterministic mapping table.
- Confirm the PLC can provide fresh timestamped sensor readings before each
  15-minute AI decision.
- Confirm the PLC can execute `mode 3 + PP + power 0 + flow 50%` for about
  25 seconds before the PC reads fresh data.
- Confirm normal standby should be `mode 3 + physical PP + power 0 + flow 0%`.
- Confirm `mode 4` is reserved for explicit shutdown / motor-off behavior, not
  routine standby.
- Confirm zero-power commands with physical PP, such as `01,0,0,` and
  `01,0,50,`, are accepted and do not require PP `00`.
- Confirm command timestamp / stale-command handling on the PLC side.
- Confirm the maximum valid command ranges for power and flow.
- Confirm sign semantics: the current `Command.txt` uses non-negative
  `power_mW`; the situation / mode code determines charge / discharge meaning.
- Confirm which sensor is the best source for PV support, grid draw, load power,
  and battery voltage.
- Confirm whether grid and PV can be measured simultaneously during mixed
  support.
- Confirm how PLC safety interlocks override or reject PC commands.
- Confirm whether the PLC can report command acceptance / rejection / active
  mode feedback back to the PC.

### Open Questions for Antoine / Vendor

1. What final interface should be used: `Data.txt` / `Command.txt`, PLC memory
   registers, Modbus, OPC UA, TCP, serial, or another protocol?
2. Can the PLC provide the complete preferred input set, especially load power,
   PV / bus power, grid power, battery voltage, battery current, SoC, and flow
   feedback?
3. What is the authoritative measurement for PV support to load? We should not
   infer PV sufficiency only from `grid == 0` or a simple voltage comparison.
4. Can the PLC support the pre-measure sequence with 50% flow for about
   25 seconds before a fresh data read?
5. Should the PC command refresh rate between 15-minute decisions remain about
   10 seconds, or should the PLC latch the last valid command until the next
   timestamped command?
6. What is the PLC-side timeout if PC commands stop arriving?
7. Are there vendor-side hard limits for pump flow, battery charge power,
   battery discharge power, and mode transitions?
8. Does `mode 4` currently shut off the motor / battery path exactly as assumed?
9. Can the PLC return an acknowledgement or actual active mode after each
   command?
10. What test cases does Antoine want for July integration acceptance?

### Suggested Integration Schedule / Test Plan

#### July: Establish Control Connection

- Define the exact I/O protocol and field mapping.
- Implement or confirm `Data.txt` / `Command.txt` compatibility.
- Run PC AI in dry-run mode while the PLC / vendor controller produces live
  data.
- Verify command parsing without energizing risky actions.
- Test standby command: `mode 3 + PP + 0 mW + 0% flow`.
- Test pre-measure command: `mode 3 + PP + 0 mW + 50% flow` for about
  25 seconds.
- Test charge / discharge command acceptance with conservative limits.
- Test PLC rejection / fallback behavior for invalid commands.

#### August: Validate and Optimize

- Run controlled short trials under supervision.
- Compare sensor logs, PC raw model action, guarded action, and PLC active mode.
- Validate voltage cutoff and recovery behavior.
- Validate PV-active discharge blocking.
- Validate SoC guard behavior.
- Validate command timestamp / timeout behavior.
- Collect flow / motor-power data if flow optimization will be considered.
- Only after stable short tests, run longer multi-window validation.
