# Installation Guide

## Prerequisites

- Python 3.8+ (tested with 3.11)
- conda (recommended) or pip
- Git

## Option A: Conda (Recommended)

```bash
# Create and activate environment
conda create -n rose python=3.11 -y
conda activate rose

# Install core dependencies
conda install numpy scipy matplotlib pandas scikit-learn -y
pip install pysindy statsmodels python-dateutil
```

## Option B: pip + venv

```bash
python -m venv rose_env
source rose_env/bin/activate   # Linux/macOS
# rose_env\Scripts\activate    # Windows

pip install numpy scipy matplotlib pandas scikit-learn
pip install pysindy statsmodels python-dateutil
```

## Raspberry Pi (Stages 4-5 only)

For hardware MPC and calibration scripts running on the Raspberry Pi:

```bash
# GPIO and I2C
pip install RPi.GPIO smbus2

# SPI (for MCP3008 ADC)
pip install spidev

# Adafruit sensor libraries
pip install adafruit-blinka adafruit-circuitpython-vl6180x board busio

# IO Expander
pip install adafruit-circuitpython-mcp230xx
```

## Clone and Run

```bash
git clone https://github.com/bhattner143/SINDYc_MPC_RoSE_symmetric_peristaltic.git
cd SINDYc_MPC_RoSE_symmetric_peristaltic

# Verify installation: train a model
conda activate rose
python 1_sindy_model_training/ADC/RoSEv2pt0_ADC_SINDYc_using_PySINDY_discrete_states_1_symmetric_07_10.py
```

## Dependencies Summary

| Package | Version | Purpose | Required For |
|---------|---------|---------|--------------|
| `numpy` | >= 1.20 | Array operations | All stages |
| `scipy` | >= 1.7 | Optimization, signal processing, interpolation | All stages |
| `matplotlib` | >= 3.3 | Plotting and visualization | All stages |
| `pandas` | >= 1.3 | CSV data loading and manipulation | All stages |
| `pysindy` | >= 1.7 | Sparse identification (SINDYc) | Stage 1, 3 |
| `scikit-learn` | >= 0.24 | Linear regression, preprocessing | Stage 1 |
| `statsmodels` | >= 0.12 | Statistical tests (ADF, KPSS) | Stage 1 |
| `python-dateutil` | >= 2.8 | Timestamp parsing from CSV data | Stage 1 |
| `RPi.GPIO` | latest | Raspberry Pi GPIO control | Stage 4, 5 |
| `spidev` | latest | SPI interface (MCP3008 ADC) | Stage 4, 5 |
| `smbus2` | latest | I2C interface (VL6180X TOF) | Stage 4, 5 |
| `adafruit-blinka` | latest | Adafruit hardware abstraction | Stage 4, 5 |
| `adafruit-circuitpython-vl6180x` | latest | VL6180X TOF sensor driver | Stage 4, 5 |

## Troubleshooting

**ImportError: cannot import name 'zeros' from 'scipy'**
- This was a legacy import. Recent scipy versions removed `scipy.zeros`. The code now uses `numpy.zeros` instead.

**ModuleNotFoundError: No module named 'tvregdiff_master'**
- Ensure you run scripts from the project root directory, or that your IDE sets the working directory to the project root.

**RPi.GPIO / spidev import errors on desktop**
- These are Raspberry Pi-only libraries. Scripts in stages 1-3 do not require them. Hardware scripts (stages 4-5) must run on the Pi.
