# krasnov.py
#
# Usage: python krasnov.py
#
# Requires: pandas, numpy, fastprop, rdkit, thermo
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV file from Zenodo:
# https://zenodo.org/records/6984601
#
# Running this will then csv files for fastprop predicting.
from math import log10
from pathlib import Path

import pandas as pd
from thermo.chemical import Chemical

from utils import get_descs, drop_overlap, DROP_WATER


bigsol_data: pd.DataFrame = pd.read_csv("BigSolDB.csv")
print(len(bigsol_data), "<--- number of samples in the original dataset")
bigsol_data = bigsol_data[~bigsol_data["Solvent"].isin(("PEG-400", "PEG-300", "PEG-200"))]
if DROP_WATER:
    bigsol_data = bigsol_data[~bigsol_data["Solvent"].isin(("water", ))]
print(len(bigsol_data), "<--- number of samples without PEG")


# convert the mol fraction to concentration
def _fraction_to_molarity(row):
    name = row["Solvent"]
    if name == "THF":
        name = "tetrahydrofuran"
    elif name == "n-heptane":
        name = "heptane"
    elif name == "DMS":
        name = "methylthiomethane"
    elif name == "2-ethyl-n-hexanol":
        name = "2-Ethyl hexanol"
    elif name == "3,6-dioxa-1-decanol":
        name = "butoxyethoxyethanol"
    elif name == "DEF":
        name = "diethylformamide"
    try:
        m = Chemical(name, T=row["T,K"])
    except ValueError:
        print(f"Could not find chemical name {name}.")
        return pd.NA
    try:
        return log10(row["Solubility"] / (m.MW / m.rho))
    except TypeError as e:
        print(name, "could not be estimated.")
        print(str(e))
        return pd.NA


bigsol_data.insert(1, "logS", bigsol_data[["T,K", "Solvent", "Solubility"]].apply(_fraction_to_molarity, axis=1))
bigsol_data = bigsol_data.dropna()
print(len(bigsol_data), "<-- size without un-estimable solvents")
bigsol_data = bigsol_data.rename(columns={"T,K": "temperature", "SMILES": "solute_smiles", "SMILES_Solvent": "solvent_smiles", "Source": "source"})


# drop multiple-fragment species
def _is_one_mol(row):
    if "." in row["solute_smiles"]:
        return False
    return True


bigsol_data = bigsol_data[bigsol_data[["solute_smiles"]].apply(_is_one_mol, axis=1)]
print(len(bigsol_data), "<-- count after dropping non-single molecule solutes")
bigsol_data = bigsol_data.reset_index()
sources = bigsol_data["source"].to_list()

fastprop_data = get_descs(bigsol_data)
fastprop_data.insert(1, "source", sources)
fastprop_data = drop_overlap(fastprop_data, "vermeire")

_dest = Path("krasnov")
if not Path.exists(_dest):
    Path.mkdir(_dest)
fastprop_data.to_csv(_dest / (f"bigsoldb_fastprop{'_nonaq' if DROP_WATER else ''}.csv"))
fastprop_data[["logS", "temperature", "solvent_smiles", "solute_smiles", "source"]].to_csv(
    _dest / f"bigsoldb_chemprop{'_nonaq' if DROP_WATER else ''}.csv"
)
