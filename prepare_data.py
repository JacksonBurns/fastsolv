# prepare_data.py
#
# Usage: python prepare_data.py
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
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem

# load the two datafiles and concatenate them
_src_dir: str = os.path.join("SolProp_v1.2", "Data")
room_T_data: pd.DataFrame = pd.read_csv(os.path.join(_src_dir, "CombiSolu-Exp-HighT.csv"))
high_T_data: pd.DataFrame = pd.read_csv(os.path.join(_src_dir, "CombiSolu-Exp.csv"))
all_data: pd.DataFrame = pd.concat((room_T_data, high_T_data))
# drop those missing the solubility
all_data: pd.DataFrame = all_data[all_data["experimental_logS [mol/L]"].notna()].reset_index()

# find all the unique molecules in the dataset and calculate their descriptors
unique_smiles: np.ndarray = np.hstack((pd.unique(all_data["solvent_smiles"]), pd.unique(all_data["solute_smiles"])))
descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)

# assemble the data into the format expected in fastprop
# map smiles -> descriptors
smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
fastprop_data: pd.DataFrame = all_data[["solute_smiles", "solvent_smiles", "experimental_logS [mol/L]", "temperature"]].rename(
    columns={"experimental_logS [mol/L]": "logS"}
)
descriptor_columns: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]
fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + descriptor_columns)
fastprop_data[descriptor_columns] = [
    np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
    for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
]
fastprop_data.to_csv(Path("models/fastprop_custom/prepared_data.csv"))
fastprop_data[["temperature"] + descriptor_columns].to_csv(Path("models/fastprop_plain/features.csv"))
fastprop_data[["solute_smiles", "solvent_smiles", "logS"]].to_csv(Path("models/fastprop_plain/targets.csv"))
fastprop_data[["temperature"]].to_csv(Path("models/chemprop/chemprop_features.csv"))
