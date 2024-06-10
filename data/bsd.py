# bsd.py
#
# Usage: python bsd.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV file at:
# https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/main/solubility/biogen_solubility.csv
#
# Running this will then csv files for fastprop predicting.
from pathlib import Path

import pandas as pd

from utils import get_descs

_dest = Path("bsd")
if not Path.exists(_dest):
    Path.mkdir(_dest)

df = pd.read_csv("biogen_solubility.csv")
df.insert(0, "temperature", 273.15 + 25)
df.insert(0, "solvent_smiles", "O")
df = df.rename(columns={"SMILES": "solute_smiles"})

fastprop_data = get_descs(df)

fastprop_data.to_csv(_dest / "bsd_features.csv")
