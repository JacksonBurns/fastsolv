# `fastsolv`
This directory contains the `fastsolv` python package which allows using the trained `fastsolv` model for solid solubility prediction.

Run `pip install fastsolv` to install it.
Trained model checkpoints will be auto-magically downloaded on your first run of `fastsolv`.

`fastsolv` is accessible via the command line and as a python module.
 - command line: run `fastsolv --help` for usage instructions
 - python module: import the `fastsolv` predictor with `from fastsolv import fastsolv` - predictions can then be made by passing a `pandas.DataFrame` with the columns for `solute_smiles`, `solvent_smiles`, and `temperature`.

To manually load `fastsolv` models and make predictions using `torch` on your own, adapt the code in `fastsolv._module`.
