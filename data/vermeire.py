# vermeire.py
#
# Usage: python vermeire.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the dataset from Zenodo:
# https://zenodo.org/records/5970538/files/SolProp_v1.2.zip?download=1
# and decompressing it in this directory.
#
# Running this will then csv files for fastprop training.
from pathlib import Path

import pandas as pd

from utils import get_descs, DROP_WATER

# load the two datafiles and concatenate them
_src_dir: str = Path("SolProp_v1.2/Data")
room_T_data: pd.DataFrame = pd.read_csv(_src_dir / "CombiSolu-Exp-HighT.csv")
high_T_data: pd.DataFrame = pd.read_csv(_src_dir / "CombiSolu-Exp.csv")
all_data: pd.DataFrame = pd.concat((room_T_data, high_T_data))
# drop those missing the solubility
all_data: pd.DataFrame = all_data[all_data["experimental_logS [mol/L]"].notna()].reset_index()
# rename columns
all_data = all_data.rename(columns={"experimental_logS [mol/L]": "logS"})

fastprop_data: pd.DataFrame = get_descs(all_data)

_dest = Path("vermeire")
if not Path.exists(_dest):
    Path.mkdir(_dest)
fastprop_data.insert(0, "source", all_data["source"])

if DROP_WATER:
    fastprop_data = fastprop_data[~fastprop_data["solvent_smiles"].eq("O")].reset_index(drop=True)

fastprop_data[["solute_smiles", "solvent_smiles", "temperature", "logS"]].to_csv(
    _dest / f"solprop_chemprop{'_nonaq' if DROP_WATER else ''}.csv",
)
fastprop_data.to_csv(_dest / f"solprop_fastprop{'_nonaq' if DROP_WATER else ''}.csv")
