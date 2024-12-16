# `fastsolv`
This directory contains the `fastsolv` python package which allows using the trained `fastsolv` model for solid solubility prediction.

## Installation

Run `pip install fastsolv` to install it from PyPI.
Trained model checkpoints will be auto-magically downloaded on your first run of `fastsolv`.

### Requirements
`fastsolv` is continually tested on all platforms (Windows, MacOS, Linux) with Python versions 3.8 and newer and the latest dependencies.
A Graphics Processing Unit (GPU) is optional, but highly recommended for fast predictions.
Dependencies are automatically installed when `fastsolv` is installed with `pip` - they are `fastprop`, `torch`, `pandas`, and `numpy`.

## Usage

`fastsolv` is accessible via the command line and as a python module.
 - command line: run `fastsolv --help` for usage instructions or `fastsolv demo.csv` to run a demo (prints the predicted solubility for aspirin at various temperatures in various solvents, runs in <1 minute + 1st run checkpoint downloading time).
 - python module: import the `fastsolv` predictor with `from fastsolv import fastsolv` - predictions can then be made by passing a `pandas.DataFrame` with the columns for `solute_smiles`, `solvent_smiles`, and `temperature` (see `demo.ipynb` which runs the same predictions as the command line call above).

The CSV files generated as part of the `paper` directory may also be passed into this predictor.

To manually load `fastsolv` models and make predictions using `torch` on your own, adapt the code in `fastsolv._module`.
