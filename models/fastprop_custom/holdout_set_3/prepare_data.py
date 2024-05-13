# prepare_data.py
#
# Usage: python prepare_data.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV file from Zenodo:
# https://zenodo.org/records/6984601
#
# Running this will then csv files for fastprop predicting.
from pathlib import Path

import numpy as np
import pandas as pd
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem

bigsol_data: pd.DataFrame = pd.read_csv("BigSolDB.csv")
print(len(bigsol_data), "<--- number of molecules in the original dataset")
print("Dropping PEG")
bigsol_data = bigsol_data[~bigsol_data["Solvent"].isin(("PEG-400", "PEG-300", "PEG-200"))]
print(len(bigsol_data), "<--- number of molecules without PEG")
# drop any which are also in our data
vermeire_smiles = set(Chem.CanonSmiles(s) for s in pd.read_csv("../prepared_data.csv", index_col=0)["solute_smiles"])
bigsol_smiles = set(Chem.CanonSmiles(s) for s in bigsol_data["SMILES"])
overlapped_smiles = tuple(vermeire_smiles.intersection(bigsol_smiles))

bigsol_data = bigsol_data[~bigsol_data["SMILES"].isin(overlapped_smiles)]
print(len(bigsol_data), "<--- number of molecules after dropping solutes in our training dataset")

unique_smiles: np.ndarray = np.hstack((pd.unique(bigsol_data["SMILES_Solvent"]), pd.unique(bigsol_data["SMILES"])))
descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)

# assemble the data into the format expected in fastprop
# map smiles -> descriptors
smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
fastprop_data: pd.DataFrame = bigsol_data[["SMILES", "SMILES_Solvent", "Solubility", "T,K"]].rename(columns={"Solubility": "logS", "T,K": "temperature", "SMILES": "solute_smiles", "SMILES_Solvent": "solvent_smiles"})
fastprop_data["logS"] = fastprop_data["logS"].apply(np.log10)
descriptor_columns: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]
fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + descriptor_columns)
fastprop_data[descriptor_columns] = [
    np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
    for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
]
fastprop_data.to_csv("bigsol_features.csv")
