# prepare_data.py
#
# Usage: python prepare_data.py
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

import numpy as np
import pandas as pd
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem

llompart_data: pd.DataFrame = pd.read_csv("OChemUnseen.csv")
print(len(llompart_data), "<--- number of molecules in the original dataset")
# drop any which are also in our data
vermeire_smiles = set(Chem.CanonSmiles(s) for s in pd.read_csv("../prepared_data.csv", index_col=0)["solute_smiles"])
llompart_smiles = set(Chem.CanonSmiles(s) for s in llompart_data["SMILES"])
overlapped_smiles = tuple(vermeire_smiles.intersection(llompart_smiles))
# ends up removing these six molecules
# {
#     "N#Cc1c(Cl)c(Cl)c(Cl)c(C#N)c1Cl",
#     "CCOC(=O)c1ccc(O)cc1",
#     "O=C(c1ccccc1)C(O)c1ccccc1",
#     "CC(O)(CS(=O)(=O)c1ccc(F)cc1)C(=O)Nc1ccc(C#N)c(C(F)(F)F)c1",
#     "c1ccc2[nH]c(-c3cscn3)nc2c1",
#     "OC1C(O)C(O)C(O)C(O)C1O",
# }

llompart_data = llompart_data[~llompart_data["SMILES"].isin(overlapped_smiles)]
print(len(llompart_data), "<--- number of molecules after dropping those in our training dataset")

unique_smiles: np.ndarray = np.hstack(("O", pd.unique(llompart_data["SMILES"])))
descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)

# assemble the data into the format expected in fastprop
# map smiles -> descriptors
smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
fastprop_data: pd.DataFrame = llompart_data[["SMILES", "LogS"]].rename(columns={"LogS": "logS", "T": "temperature", "SMILES": "solute_smiles"})
fastprop_data.insert(0, "solvent_smiles", "O")
descriptor_columns: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]
fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + descriptor_columns)
fastprop_data[descriptor_columns] = [
    np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
    for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
]
fastprop_data.insert(0, "temperature", 273.15 + 25)
fastprop_data.to_csv("llompart_features.csv")
