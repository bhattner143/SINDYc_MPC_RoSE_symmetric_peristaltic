# SINDYc-MPC RoSE — Script Categories

Scripts are organised into 5 folders based on whether they require physical hardware.

---

## Folder Summary

| Folder | Hardware Needed? | Description |
|--------|-----------------|-------------|
| [`1_offline_model_as_plant/`](1_offline_model_as_plant/README.md) | No | Closed-loop SINDYc-MPC simulation using the identified model as the plant |
| [`2_offline_SINDYc_analysis/`](2_offline_SINDYc_analysis/README.md) | No | SINDYc system identification from CSV data, trajectory generation, data analysis (formerly `pysindy_training/`) |
| [`3_hardware_real_plant/`](3_hardware_real_plant/README.md) | **Yes (RPi)** | SINDYc-MPC running on the real RoSE robot (ADC or TOF sensing) (formerly `scripts_hardware/`) |
| [`4_hardware_calibration/`](4_hardware_calibration/README.md) | **Yes (RPi)** | One-off sensor calibration scripts (MCP3008 ADC, VL6180X TOF) (formerly `hardware_calibration/`) |
| [`5_hybrid/`](5_hybrid/README.md) | Partial | Script with `model_as_plant` name that also imports hardware — offline by default (pass `UseIOExpander=True` for hardware path) |

---

## Root-level library files (stay in root — imported by scripts in all folders)

| File | Role |
|------|------|
| `sindybase.py` | SINDy base class |
| `ClassTOFandWebCam.py` | TOF sensor + webcam data utility class |
| `ClassStentForTOF.py` | Stent / geometry utilities for TOF |
| `Class_SINDYc_MPC_Design.py` | Alternative MPC design class (not active in most scripts) |
| `QSR_TwoLayer_IOEXpander_TOF_OOPs.py` | Hardware driver — QSR 2-layer, I/O Expander + TOF/SPI (RPi only) |
| `QSR_TwoLayer_IOEXpander_TOF_six_OOPs.py` | Hardware driver — 6-segment variant (RPi only) |
| `RoSE_IOEXpander_TOF_ten_OOPs.py` | Hardware driver — RoSEv2.0 (10 sensors, RPi only) |
| `RoSE_IOEXpander_TOF_cal_ten_OOPs.py` | Hardware driver — RoSEv2.0 with calibration (RPi only) |
| `tvregdiff_master/` | TV regularised differentiation package |
| `CustomizedLibrary/` | Additional sensor classes (VL6180X, error definitions, etc.) |

---

## Import path
All scripts in sub-folders automatically insert the project root into `sys.path`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This ensures root-level libraries are found regardless of which folder the script is run from.
