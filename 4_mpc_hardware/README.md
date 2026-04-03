# Stage 4 — MPC on Hardware

Deploys MPC on Raspberry Pi to control the real RoSE v2.0 robot (Sec VI–VII).
**Requires RPi with GPIO, I2C, SPI connected to RoSE.**

## Scripts

| Script | Sensor | States | Actuation |
|--------|--------|--------|-----------|
| `*_layer2_states_1_symmetric_07_10.py` | ADC | 1 | Symmetric |
| `*_layer2_states_1_symmetric_10_10.py` | TOF | 1 | Symmetric |
| `*_states_3_peristalsis_02_11_20.py` | ADC | 3 | Peristaltic wave |
| `*_states_3_peristalsis_20_10_20.py` | ADC | 3 | Peristaltic (Oct data) |
| `*_states_3_peristalsis_04_11_20.py` | TOF | 3 | Peristaltic (Nov data) |
| `*_error_corr_Performance_eval_17_11_20.py` | TOF | 3 | Peristalsis + error correction |
| `QSR/*.py` | Both | 2 | Legacy QSR 2-layer hardware MPC |
