# Stage 1 — DTSINDYc Model Training

Trains Discrete-Time SINDYc (DTSINDYc) models from open-loop sensor data.
Discovers sparse difference equations **x(k+1) = f(x(k), u(k))** using PySINDy.

## Sub-folders

- **TOF/** — Model M1: displacement-based (VL6180X time-of-flight sensors)
- **ADC/** — Model M2: pressure-based (MCP3008 ADC / VPS sensors)
- **QSR/** — Legacy QSR 2-layer platform models

## Scripts

| Script | Model | Actuation | Data Source |
|--------|-------|-----------|-------------|
| `TOF/..._states_1_symmetric_10_10.py` | M1 | 1-state symmetric | `CsvData29_09_2020/` |
| `TOF/..._states_3_peristalsis_04_11_20.py` | M1 | 3-state peristalsis | `CsvData04_11_2020/` |
| `ADC/..._states_1_symmetric_07_10.py` | M2 | 1-state symmetric | `CsvData29_09_2020/` |
| `ADC/..._states_3_peristalsis_20_10_20.py` | M2 | 3-state peristalsis | `DataFiles/` |
| `TOF/..._exec_time_and_error_tracking.py` | M1 | Benchmark | MPC execution time & model error |

## Usage

```bash
python 1_sindy_model_training/ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_1_symmetric_07_10.py
```
