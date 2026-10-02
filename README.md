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

## Precision Support

### Vitis HLS regression and classification on AXI4-Stream

For regression (one or several targets) and single-label classification,
`--backend vitis_hls` now selects a
primitive-ROM AXI implementation by default. Fixed-point models store tree
thresholds and leaves as signed integer arrays; float models use primitive
float arrays. This avoids the constant-zero RTL produced by the earlier
aggregate tree representation in Vitis HLS 2024.1. The generated testbench
checks fixed-point prediction codes exactly and float predictions to within
`1e-5`; it checks `TLAST`, `TKEEP`, and `TSTRB` exactly in C and RTL
co-simulation. A mismatch makes the HLS run fail.

```bash
cambium quick-start --data data.csv --features f1,f2 --targets y \
  --backend vitis_hls --precision 10.4 --implementation integer_axi \
  --output forest_hls
cd forest_hls
vitis_hls -f cambium_project.tcl
```

The input is one 32-bit AXI4-Stream word per feature. Its low `W` bits contain
a signed two's-complement fixed-point integer code for `ap_fixed<W,I>`; the
remaining bits are ignored. Scale each already-preprocessed feature by
`2^(W-I)` and round down before packing. Assert `TLAST` on the final feature
of the final sample in a DMA batch. Each regression prediction or classification
score is returned as one 32-bit word with the signed result code in the low
`W` bits; decode it, divide by `2^(W-I)`, and undo target preprocessing on the
host if needed. Classification returns one score per class, in
`model.classes_` order. Float precision uses IEEE-754 binary32 words on input
and output instead. Both paths produce an IP-catalog export for DMA integration.

`--implementation auto` is the default; `--implementation struct` selects the
older exporter. The integer path supports regression or classification and up
to 32 total fixed-point bits. Vitis HLS 2024.1 C/RTL co-simulation has validated
single-output regression and classification at `ap_fixed<10,4>` and float,
regression at `ap_fixed<18,8>`, classification at `ap_fixed<16,6>`, and
two-output regression at `ap_fixed<10,4>` and float. Other precisions and boards
require their own verification before deployment. The existing Vivado HLS
export path remains available; Vivado HLS itself was not installed in this
validation environment.

Tree count (`--n-estimators`) and maximum depth (`--max-depth`) control the
trained model. `--max-leaf-nodes` caps the number of leaves per trained tree.
`--max-nodes` sets the minimum HLS array capacity; Cambium increases it when
the trained tree needs more nodes, so it is not a training limit.

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
- `vivado_block_design.tcl`

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

```bash
cd energy_pred
vivado -mode batch -source vivado_block_design.tcl
```

### Vitis HLS (optional)

```bash
python cambium_cli.py quick-start --data data.csv --features f1,f2 --targets y --backend vitis_hls --output out_vitis --precision 18.8
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
