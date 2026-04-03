# Stage 2 — Reference Trajectory Generation

Generates peristaltic wave reference signals for MPC tracking (Eq. 14 in paper).
Defines the desired displacement/pressure profile for each RoSE layer over time.

## Scripts

| Script | Domain | Output |
|--------|--------|--------|
| `RoSE_MPC_tof_trajectory_gen.py` | TOF (displacement, mm) | Reference CSV for TOF-based MPC |
| `RoSE_MPC_adc_trajectory_gen.py` | ADC (pressure, mV) | Reference CSV for ADC-based MPC |

Generated trajectories are saved to `ControllerReferenceFiles/`.

## Usage

```bash
python 2_trajectory_generation/RoSE_MPC_tof_trajectory_gen.py
```
