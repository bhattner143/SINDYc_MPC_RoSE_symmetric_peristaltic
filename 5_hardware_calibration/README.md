# Stage 5 — Hardware Calibration

Sensor calibration routines. Run these **before** deploying MPC on hardware (Stage 4).
**Requires Raspberry Pi.**

## Scripts

| Script | Purpose |
|--------|---------|
| `RoSE_TOF_offset_crostalk_convtime_calibration.py` | TOF offset, crosstalk & convergence-time calibration |
| `Mcp3008_FSP_Cal.py` | MCP3008 ADC full-scale pressure calibration |
| `test_mcp3008_spidev.py` | Quick SPI connectivity test for MCP3008 |

Calibration output files are saved to `TOFCalFiles/`.
