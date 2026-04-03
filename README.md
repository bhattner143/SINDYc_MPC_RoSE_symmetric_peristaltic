# SINDYc-MPC for RoSE Peristaltic Soft Robot

Discrete-Time Sparse Identification of Nonlinear Dynamics with control (**DTSINDYc**)
combined with Model Predictive Control (**MPC**) for the **RoSE v2.0**
(Robot of Soft Elastomer). The project implements the full pipeline from
sensor data to model identification to trajectory generation to MPC control,
as described in our IEEE paper.

System identification uses [PySINDy](https://github.com/dynamicslab/pysindy).

# Main MPC script used in the paper

The following script in the root folder can be run for Offline MPC simulation (model as plant, no hardware)

script_RoSEv2pt0_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant_states_3_peristalsis_20_10_20.py

---

## Pipeline Overview

The project follows a 5-stage pipeline. Each numbered folder corresponds to
one stage --- run them in order:

```
 +-------------------------+
 |  Experimental Data      |  csv_data/, DataFiles/
 |  (open-loop actuation)  |  CsvData22_07_2020, CsvData29_09_2020, CsvData04_11_2020
 +-----------+-------------+
             v
 +-------------------------+
 | 1. Model Training       |  1_sindy_model_training/
 |    DTSINDYc (Sec IV)    |  Trains Model M1 (TOF) and M2 (ADC/VPS)
 +-----------+-------------+
             v
 +-------------------------+
 | 2. Trajectory Gen       |  2_trajectory_generation/
 |    (Sec VI, Eq. 14)     |  Generates peristaltic reference signals
 +-----------+-------------+
             v
 +-------------------------+
 | 3. MPC Simulation       |  3_mpc_simulation/
 |    (Sec V)              |  Offline MPC -- identified model used as plant
 +-----------+-------------+
             v
 +-------------------------+
 | 4. MPC on Hardware      |  4_mpc_hardware/
 |    (Sec VI-VII)         |  Real-time MPC on RPi controlling RoSE v2.0
 +-----------+-------------+
             |
 +-------------------------+
 | 5. Calibration          |  5_hardware_calibration/
 |    (Sec III)            |  TOF & ADC sensor calibration (prerequisite)
 +-------------------------+
```

---

## Repository Structure

```
SINDYc_MPC_RoSE_symmetric_peristaltic/
|
+-- 1_sindy_model_training/       # DTSINDYc model identification from sensor data
|   +-- TOF/                      # Model M1 -- displacement (VL6180X TOF sensors)
|   +-- ADC/                      # Model M2 -- pressure (MCP3008 ADC / VPS sensors)
|   +-- QSR/                      # Legacy QSR platform models
|
+-- 2_trajectory_generation/      # Peristaltic reference trajectory generation (Eq. 14)
|
+-- 3_mpc_simulation/             # Offline MPC simulation (model as plant, no hardware)
|   +-- QSR/                      # Legacy QSR MPC simulations
|
+-- 4_mpc_hardware/               # MPC deployed on Raspberry Pi with real RoSE
|   +-- QSR/                      # Legacy QSR hardware MPC
|
+-- 5_hardware_calibration/       # Sensor calibration scripts (RPi required)
|
+-- lib/                          # Shared library modules
|   +-- sindybase.py              # SINDy base class (polynomial expansion, STLSQ)
|   +-- ClassTOFandWebCam.py      # TOF sensor + webcam data utility class
|   +-- ClassStentForTOF.py       # Stent geometry utilities for TOF
|   +-- Class_SINDYc_MPC_Design.py # MPC design class
|   +-- RoSE_IOEXpander_TOF_ten_OOPs.py       # Hardware driver (RPi only)
|   +-- RoSE_IOEXpander_TOF_cal_ten_OOPs.py   # Hardware driver with calibration (RPi only)
|
+-- csv_data/                     # Recorded experimental CSV datasets
|   +-- CsvData22_07_2020/        # July 2020: symmetric actuation data
|   +-- CsvData29_09_2020/        # Sept 2020: updated sensor data
|   +-- CsvData04_11_2020/        # Nov 2020: peristalsis + TOF data
+-- ControllerReferenceFiles/     # Pre-computed MPC reference trajectories
+-- PeristalsisData/              # Peristalsis wave DAC command tables (see PeristalsisData/README.md)
+-- DataFiles/                    # Symlink -> data_files/ (experimental data root)
+-- CustomizedLibrary/            # Sensor driver classes (VL6180X, error defs)
+-- TOFCalFiles/                  # TOF calibration parameter files
+-- tvregdiff_master/             # TV regularised differentiation library
+-- archives/                     # Legacy / superseded scripts
+-- notes/                        # LaTeX documentation with equations and diagrams
|
+-- README.md                     # This file
+-- INSTALL.md                    # Installation instructions
+-- SCRIPT_CATEGORIES.md          # Detailed per-script categorisation table
```

---

## Stage 1 -- DTSINDYc Model Training (`1_sindy_model_training/`)

Trains discrete-time SINDYc models from open-loop experimental sensor data.
Two sensor modalities produce two complementary models:

| Sub-folder | Model | Sensor | Data | Description |
|------------|-------|--------|------|-------------|
| `TOF/` | **M1** | VL6180X TOF (displacement) | `CsvData04_11_2020/`, `CsvData29_09_2020/` | Identifies state-space model from displacement measurements of layers L5, L6, L7 |
| `ADC/` | **M2** | MCP3008 ADC (pressure/VPS) | `CsvData29_09_2020/`, `DataFiles/` | Identifies state-space model from pressure measurements |
| `QSR/` | Legacy | Both | Earlier datasets | QSR 2-layer platform models (continuous & discrete) |

**Key scripts:**
- `TOF/RoSEv2pt0_TOF_SINDYc_using_PySINDY_discrete_states_1_symmetric_10_10.py` -- M1, 1-state symmetric actuation
- `TOF/RoSEv2pt0_TOF_SINDYc_using_PySINDY_discrete_states_3_peristalsis_04_11_20.py` -- M1, 3-state peristalsis
- `ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_1_symmetric_07_10.py` -- M2, 1-state symmetric
- `ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_3_peristalsis_20_10_20.py` -- M2, 3-state peristalsis
- `TOF/RoSEv2pt0_TOF_SINDYc_discrete_MPC_exec_time_and_error_tracking.py` -- Benchmarks MPC execution time and model error

**How it works:** Each script loads CSV sensor data, constructs state/control matrices,
applies PySINDy (with `sindybase.py` providing polynomial library expansion and STLSQ),
and discovers sparse difference equations of the form: **x(k+1) = f(x(k), u(k))**.

```bash
python 1_sindy_model_training/ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_1_symmetric_07_10.py
```

---

## Stage 2 -- Reference Trajectory Generation (`2_trajectory_generation/`)

Generates peristaltic wave reference signals for MPC tracking. The reference
trajectory defines the desired displacement/pressure profile for each RoSE layer
over time (Eq. 14 in paper: sinusoidal peristalsis with speed 20 mm/s, wavelength 75 mm).

| Script | Sensor Domain | Output |
|--------|--------------|--------|
| `RoSE_MPC_tof_trajectory_gen.py` | TOF (displacement, mm) | Reference CSV for TOF-based MPC |
| `RoSE_MPC_adc_trajectory_gen.py` | ADC (pressure, mV) | Reference CSV for ADC-based MPC |

Generated trajectories are saved to `ControllerReferenceFiles/`.

```bash
python 2_trajectory_generation/RoSE_MPC_tof_trajectory_gen.py
```

---

## Stage 3 -- MPC Simulation (`3_mpc_simulation/`)

Runs the MPC controller offline with the **identified DTSINDYc model as the plant**
(no hardware required). This validates the controller before hardware deployment (Sec V).

The MPC optimises a cost function with prediction horizon Np=4, minimising tracking
error while respecting actuator constraints.

```bash
python 3_mpc_simulation/RoSEv2pt0_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant_states_3_peristalsis.py
```

---

## Stage 4 -- MPC on Hardware (`4_mpc_hardware/`)

Deploys MPC on **Raspberry Pi** to control the real RoSE v2.0 robot (Sec VI-VII).
**Requires RPi with GPIO, I2C, SPI connected to RoSE.**

| Script | Sensor | States | Actuation |
|--------|--------|--------|-----------|
| `*_layer2_states_1_symmetric_07_10.py` | ADC | 1 (symmetric) | All layers in unison |
| `*_layer2_states_1_symmetric_10_10.py` | TOF | 1 (symmetric) | All layers in unison |
| `*_states_3_peristalsis_*.py` | ADC/TOF | 3 (L5,L6,L7) | Peristaltic wave |
| `*_error_corr_Performance_eval_17_11_20.py` | TOF | 3 | Peristalsis + error correction |

---

## Stage 5 -- Hardware Calibration (`5_hardware_calibration/`)

Sensor calibration routines. Run these **before** hardware MPC (Stage 4).

| Script | Purpose |
|--------|---------|
| `RoSE_TOF_offset_crostalk_convtime_calibration.py` | TOF offset, crosstalk & convergence-time calibration |
| `Mcp3008_FSP_Cal.py` | MCP3008 ADC full-scale pressure calibration |
| `test_mcp3008_spidev.py` | Quick SPI connectivity test for MCP3008 |

---

## Shared Libraries (`lib/`)

All scripts import these via `sys.path.insert(0, _ROOT)` and `sys.path.insert(0, os.path.join(_ROOT, 'lib'))`:

| File | Purpose |
|------|---------|
| `sindybase.py` | SINDy base class -- polynomial library expansion, STLSQ sparse regression |
| `ClassTOFandWebCam.py` | TOF sensor data reader + webcam utility class |
| `ClassStentForTOF.py` | Stent geometry utilities for TOF sensor placement |
| `Class_SINDYc_MPC_Design.py` | MPC design class (alternative formulation) |
| `RoSE_IOEXpander_TOF_ten_OOPs.py` | RoSE hardware driver -- IO expander + 10 TOF sensors (RPi only) |
| `RoSE_IOEXpander_TOF_cal_ten_OOPs.py` | Same driver with built-in calibration (RPi only) |

---

## Peristalsis Wave Data (`PeristalsisData/`)

> **Generated by Steven Dirven** (PhD student, predecessor) as part of the open-loop
> actuation experiments on RoSE v2.0.

Pre-computed DAC command lookup tables that drive the 12 robot layers through peristaltic wave motions. Each CSV row is one time-sample; each column is a DAC command (0–255) for one layer. Baseline resting pressure = **40**; typical peak = **120**.

**See [`PeristalsisData/README.md`](PeristalsisData/README.md) for the full file inventory, naming convention, and wave timing derivation.**

### File groups at a glance

| Group | Filename pattern | Rows | Cols | Purpose |
|-------|-----------------|------|------|---------|
| Single wave cycle | `[Peristalsis_]{A}at{V}[_Dips].csv` | 148–149 | 12 (or 32) | One full peristaltic cycle at amplitude `A`, speed `V` |
| All-layer step | `Peristalsis_Staircase_0_{target}_20mmps.csv` | 1000 | 12 | Binary step 0→target, all layers simultaneously (system ID) |
| Multi-level staircase | `Peristalsis_Staircase_60_100_130_20mmps.csv` | 2500 | 12 | Three-level step sequence 60→100→130 DAC |
| 2-layer trajectory | `Peristalsis_Staircase_2layer_traj_*.csv` | 370–909 | 12 | 2-layer sub-segment validation trajectories |
| FSP calibration ramp | `FSPPeristalsisFSP_ESR_traj_40mmps_{target}.csv` | 609–809 | 12 | Uniform ramp 30→target in +2 steps (pressure-displacement calibration) |
| Parameter descriptor | `Peristalsis_Staircase_{start}_10_{end}.csv` | 1 | 12 | Single-row starting-level parameter files |

**Speed → inter-layer delay** (layer spacing = 15 mm):

| Wave speed | Inter-layer delay | Ts |
|-----------|-------------------|----|
| 20 mm/s | ~10 samples | 0.075 s |
| 30 mm/s | ~5 samples | 0.115 s |
| 40 mm/s | ~4 samples | 0.107 s |

---

## Quick Start (offline, no hardware)

```bash
conda activate rose

# Train a DTSINDYc model from recorded data
python 1_sindy_model_training/ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_1_symmetric_07_10.py

# Generate a reference trajectory
python 2_trajectory_generation/RoSE_MPC_tof_trajectory_gen.py

# Run MPC simulation (model as plant)
python 3_mpc_simulation/RoSEv2pt0_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant_states_3_peristalsis.py
```

See [INSTALL.md](INSTALL.md) for full installation instructions.
