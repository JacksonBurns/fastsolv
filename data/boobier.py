# boobier.py
#
# Usage: python boobier.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV files from GitHub:
# https://github.com/BNNLab/Solubility_data/tree/1.0/Solubility%20Data
#
# Running this will then csv files for fastprop predicting.
from pathlib import Path

import pandas as pd

from utils import drop_overlap, get_descs

_dest = Path("boobier")
if not Path.exists(_dest):
    Path.mkdir(_dest)
for src_file, solvent_smiles in zip(
    ("acetone_solubility_data.csv", "benzene_solubility_data.csv", "ethanol_solubility_data.csv"),
    ("CC(=O)C", "C1=CC=CC=C1", "OCC"),
):
    # load the two datafiles and concatenate them
    all_data: pd.DataFrame = pd.read_csv(src_file).dropna()
    # drop those missing the temperature
    all_data: pd.DataFrame = all_data[all_data["T"].notna()].reset_index()
    all_data["T"] = all_data["T"] + 273.15
    all_data = all_data.rename(columns={"LogS": "logS", "T": "temperature", "SMILES": "solute_smiles"})
    all_data.insert(0, "solvent_smiles", solvent_smiles)

    fastprop_data = get_descs(all_data)
    fastprop_data = drop_overlap(fastprop_data, "krasnov")
    fastprop_data.to_csv(_dest / ("leeds_" + Path(src_file).stem.split("_")[0] + "_fastprop.csv"))
    fastprop_data[["logS", "temperature", "solvent_smiles", "solute_smiles"]].to_csv(
        _dest / ("leeds_" + Path(src_file).stem.split("_")[0] + "_chemprop.csv")
    )
