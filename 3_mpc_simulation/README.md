# Stage 3 — MPC Simulation (Model as Plant)

Offline MPC using the identified DTSINDYc model as the simulated plant.
Validates the controller before hardware deployment (Sec V). No hardware required.

## Scripts

| Script | Sensor | States | Description |
|--------|--------|--------|-------------|
| `RoSEv2pt0_ADC_SINDYc_..._model_as_plant_states_3_peristalsis.py` | ADC | 3 | Peristaltic MPC sim |
| `script_RoSEv2pt0_ADC_SINDYc_..._model_as_plant_states_3_peristalsis_20_10_20.py` | ADC | 3 | Oct 2020 data variant |
| `QSR/*.py` | ADC | 2 | Legacy QSR 2-layer MPC simulations |

## Usage

```bash
python 3_mpc_simulation/RoSEv2pt0_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant_states_3_peristalsis.py
```
