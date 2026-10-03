# Cambium

Cambium is a Python framework that trains Random Forest models and exports FPGA-ready HLS inference code for Xilinx Vivado HLS / Vitis HLS workflows.

It is designed for fast end-to-end flow:

1. Load tabular dataset
2. Train and evaluate model
3. Export C++/HLS source + TCL scripts
4. Run synthesis and hardware integration

## Key Features

- End-to-end CLI for training + HLS export
- Random Forest regression/classification support
- Multi-output regression support
- Automatic preprocessing and scaling
- Jinja2-based hardware code generation templates
- Backend-aware export for `vivado_hls` and `vitis_hls`
- Auto-generated artifacts:
  - model/scaler/config/metrics files
  - HLS source (`.h` / `.cpp`)
  - HLS project TCL
  - Vivado block design TCL
  - Markdown report

## Repository Structure

```text
Cambium/
|-- cambium/
|   |-- __init__.py
|   |-- cli.py
|   |-- config.py
|   |-- data.py
|   |-- training.py
|   |-- codegen.py
|   `-- framework.py
|-- templates/
|   |-- firmware/
|   `-- test/
|-- cambium_cli.py          # Main CLI entrypoint
`-- README.md
```

## Requirements

- Python 3.9+ (recommended: 3.10)
- Python packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `jinja2`
  - `pyyaml`

Install quickly:

```bash
pip install numpy pandas scikit-learn jinja2 pyyaml
```

Install from PyPI:

```bash
pip install cambium-hls
```

Python import:

```python
from cambium import CambiumFramework
```
## Quick Start (Ex. Energy Dataset)

From repository root:

```bash
python cambium_cli.py quick-start --data https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp --targets Elec_Cons --output energy_pred --verbose
```

This will:

- train a model
- evaluate inference performance
- save model + metrics
- generate HLS files and TCL scripts in `energy_pred/`

## Vitis HLS on ZCU104

The following example trains a five-feature, single-output regression model,
generates an AXI4-Stream HLS project, and targets the ZCU104 device. Run it
from the repository root with the dependencies and Vitis HLS installed.

```bash
python cambium_cli.py quick-start \
  --data https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv \
  --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp \
  --targets Elec_Cons --task regression \
  --backend vitis_hls --precision 18.8 \
  --fpga-part xczu7ev-ffvc1156-2-e --clock-period 5 \
  --n-estimators 20 --max-depth 6 --output energy_pred_zcu104
cd energy_pred_zcu104
vitis_hls -f cambium_project.tcl
```

`--backend` selects `vitis_hls` or `vivado_hls`. `--precision 18.8` means
`ap_fixed<18,8>`; `--precision float` selects single-precision floating point.
The default `--implementation auto` uses Cambium's original struct-based
tree traversal and AXI4-Stream dataflow implementation.

For Vitis, the HLS testbench checks the first 32 saved holdout vectors by
default and stops at the first exact output or AXI sideband mismatch. Use
`--hls-test-samples N` to change that count. The Python metrics and saved
`X_test.npy` 

The `--fpga-part` option is important: without it, the generated project uses
the framework's default `xc7z020clg400-1`. The
`--clock-period` value is a design target in nanoseconds
clock. Change `--n-estimators`, `--max-depth`, or `--max-leaf-nodes` to control
forest size; `--max-nodes` only sets a minimum exported array capacity. For
classification, set `--task classification` and one target column. For
multi-output regression, list target columns in `--targets`.

The saved `scaler_x.pkl`, `scaler_y.pkl`, and `cambium_config.yaml` describe the
preprocessing needed by a host application. The generated testbench writes decoded simulation predictions to `Y_hls_pred.csv`.


| Option | Use |
| --- | --- |
| `--data`, `--features`, `--targets` | Select the CSV or URL, ordered input columns, and target column(s). |
| `--task` | Choose `regression` or `classification`; classification accepts one target column. |
| `--backend` | Choose `vitis_hls` or `vivado_hls`. |
| `--precision` | Choose `float`, `fixed`, or a fixed-point format such as `18.8`. |
| `--fpga-part` | Set the target device for a Vitis `quick-start` export. |
| `--clock-period` | Set the Vitis HLS clock target in nanoseconds. |
| `--n-estimators`, `--max-depth`, `--max-leaf-nodes` | Set tree count and tree complexity. |
| `--implementation` | `auto` selects the original struct core; `integer_axi` and `float_axi` are optional alternative Vitis exporters. |
| `--hls-test-samples` | Limit vectors in the Vitis struct C/RTL testbench; default 32. |
| `--output` | Choose a new directory for generated files and reports. |

Cambium supports precision overrides directly from the CLI.

Accepted forms:

- `--precision 18.8` -> `ap_fixed<18,8>`
- `--precision 16.6` -> `ap_fixed<16,6>`
- `--precision "ap_fixed<20,7>"`
- `--precision fixed`
- `--precision float`

Example:

```bash
python cambium_cli.py quick-start --data https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp --targets Elec_Cons --output energy_pred_p188 --backend vivado_hls --precision 18.8 --verbose
```
# 5. Run with the larger test set
```bash
python3 cambium_cli.py quick-start \
  --data alldata.csv \
  --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp \
  --targets Elec_Cons --task regression \
  --backend vitis_hls --precision 18.8 \
  --fpga-part xczu7ev-ffvc1156-2-e --clock-period 5 \
  --n-estimators 20 --max-depth 6 \
  --hls-test-samples 1500 \
  --output energy_pred_zcu104
```
## Main CLI Commands

### 1) Quick start

```bash
python cambium_cli.py quick-start --data data.csv --features f1,f2 --targets y --output out_dir
```

With backend and precision override:

```bash
python cambium_cli.py quick-start --data data.csv --features f1,f2 --targets y --output out_dir --backend vivado_hls --precision 16.6
```

Classification examples:

```bash
python cambium_cli.py quick-start --data iris.csv --features sepal_length,sepal_width,petal_length,petal_width --targets species --task classification --backend vitis_hls --precision 18.8 --output iris_classifier --verbose
```

```bash
python cambium_cli.py quick-start --data digits.csv --targets digit --task classification --backend vivado_hls --precision 18.8 --output digits_classifier --verbose
```

Real-world HVAC regression example:

```bash
python cambium_cli.py quick-start --data https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp --targets Elec_Cons,Therm_Eng_Cons,PMV --backend vivado_hls --precision 18.8 --output hvac_multi_target --verbose
```

### 2) Train only

```bash
python cambium_cli.py train --data data.csv --features f1,f2 --targets y --output out_dir --save-model model.pkl
```

### 3) Export from saved model

```bash
python cambium_cli.py export --model trained_dir/cambium_model.pkl --output out_dir
```

When the model directory contains `cambium_config.yaml` and `X_test.npy`, export
restores the original feature order, backend, precision, and held-out vectors.
Use `--backend`, `--precision`, or `--implementation` to override those settings.
For a model saved elsewhere, provide `--data`, `--features`, and `--targets`
(and `--task` for classification) so Cambium can build verification vectors.

### 4) Create config template

```bash
python cambium_cli.py create-config --output cambium_config.yaml
```

You can also keep precision in YAML:

```yaml
export:
  precision: "ap_fixed<18,8>"
```

### 5) Print synthesis report summary

```bash
python cambium_cli.py hls-report --project-dir energy_pred/cambium_project
```

## Generated Output Files

After a successful quick start, `energy_pred/` will include files such as:

- `cambium_model.pkl`
- `cambium_config.yaml`
- `cambium_metrics.json`
- `cambium_report.md`
- `X_test.npy`, `Y_test.npy`, `Y_pred.npy`
- generated firmware/test C++ and headers
- `<project_name>.tcl` (default: `cambium_project.tcl`)
- `vivado_block_design.tcl` for the Vivado HLS flow only (Zynq-7000 template)

## HLS / FPGA Flow

### Vivado HLS

```bash
cd energy_pred
vivado_hls -f cambium_project.tcl
```

Example generation command:

```bash
python cambium_cli.py quick-start --data https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv --features Occupancy,Rel_Hum,Room_Temp,Air_Flow_Rat,Air_Temp --targets Elec_Cons --output energy_pred_vivado --backend vivado_hls --precision 18.8 --verbose
```

### Bitstream generation (Vivado batch)

The generated `vivado_block_design.tcl` uses a Zynq-7000 processing-system
template. It is not a ZCU104 deployment script; for ZCU104, integrate the
exported IP into a Zynq UltraScale+ block design with AXI DMA first.

```bash
cd energy_pred
vivado -mode batch -source vivado_block_design.tcl
```

### Vitis HLS

```bash
python cambium_cli.py quick-start --data data.csv --features f1,f2 --targets y --backend vitis_hls --precision 18.8 --fpga-part xczu7ev-ffvc1156-2-e --output out_vitis
```

Then run:

```bash
cd out_vitis
vitis_hls -f cambium_project.tcl
```

Notes:

- Vivado HLS and Vitis HLS are both supported.
- Regenerate the project when switching backend.
- Using a fresh output directory per backend is recommended, for example `energy_pred_vivado` and `energy_pred_vitis`.

## Tutorial Notebook

For a walkthrough covering end-to-end framework usage, open:

- `Cambium_Tutorial_Regression_Iris_Digits.ipynb`

The notebook covers:

- multi-target HVAC regression
- iris classification
- digits classification
- CLI-based project generation
- HLS output validation against Python baseline predictions

## Python API Example

```python
from cambium import CambiumFramework

fw = CambiumFramework()
fw.load_data(
    "https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv",
    feature_cols=["Occupancy", "Rel_Hum", "Room_Temp", "Air_Flow_Rat", "Air_Temp"],
    target_cols=["Elec_Cons"],
)
metrics = fw.train()
fw.export_to_hls_j2("energy_pred")
fw.generate_report()
print(metrics)
```



## Author

Mohammed Mshragi
