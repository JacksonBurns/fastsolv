# prepare_data.py
#
# Usage: python prepare_data.py
#
# Requires: pandas, numpy, psutil, fastprop
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the dataset from Zenodo:
# https://zenodo.org/records/5970538/files/SolProp_v1.2.zip?download=1
# and decompressing it in this directory.
#
# Running this will then emit two csv files for fastprop training.
import os

import numpy as np
import pandas as pd
import psutil
from fastprop.utils import ALL_2D, calculate_mordred_desciptors, mordred_descriptors_from_strings
from rdkit import Chem

# load the two datafiles and concatenate them
_src_dir: str = os.path.join("SolProp_v1.2", "Data")
room_T_data: pd.DataFrame = pd.read_csv(os.path.join(_src_dir, "CombiSolu-Exp-HighT.csv"))
high_T_data: pd.DataFrame = pd.read_csv(os.path.join(_src_dir, "CombiSolu-Exp.csv"))
all_data: pd.DataFrame = pd.concat((room_T_data, high_T_data))
# drop those missing the solubility
all_data: pd.DataFrame = all_data[all_data['experimental_logS [mol/L]'].notna()]
# drop not-very-soluble species (must be at least 1 mol/L)
# all_data: pd.DataFrame = all_data[all_data['experimental_logS [mol/L]'] > 0]

# find all the unique molecules in the dataset and calculate their descriptors
unique_smiles: np.ndarray = np.hstack((pd.unique(all_data["solvent_smiles"]), pd.unique(all_data["solute_smiles"])))
descs: np.ndarray = calculate_mordred_desciptors(
    mordred_descriptors_from_strings(ALL_2D),
    list(Chem.MolFromSmiles(i) for i in unique_smiles),
    psutil.cpu_count(logical=False),
)

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
fastprop_data[["temperature"] + descriptor_columns].to_csv("features.csv")
fastprop_data[["temperature"]].to_csv("chemprop_features.csv")
fastprop_data[["solute_smiles", "solvent_smiles", "logS"]].to_csv("targets.csv")
