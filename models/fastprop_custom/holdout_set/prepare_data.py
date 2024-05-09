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

for src_file, solvent_smiles in zip(
    ("acetone_solubility_data.csv", "benzene_solubility_data.csv", "ethanol_solubility_data.csv"),
    ("CC(=O)C", "C1=CC=CC=C1", "OCC"),
):
    # load the two datafiles and concatenate them
    all_data: pd.DataFrame = pd.read_csv(src_file).dropna()
    # drop those missing the temperature
    all_data: pd.DataFrame = all_data[all_data["T"].notna()].reset_index()
    all_data["T"] = all_data["T"] + 273.15

    # find all the unique molecules in the dataset and calculate their descriptors
    unique_smiles: np.ndarray = np.hstack((solvent_smiles, pd.unique(all_data["SMILES"])))
    descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)

    # assemble the data into the format expected in fastprop
    # map smiles -> descriptors
    smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
    fastprop_data: pd.DataFrame = all_data[["SMILES", "T", "LogS"]].rename(columns={"LogS": "logS", "T": "temperature", "SMILES": "solute_smiles"})
    fastprop_data.insert(0, "solvent_smiles", solvent_smiles)
    descriptor_columns: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]
    fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + descriptor_columns)
    fastprop_data[descriptor_columns] = [
        np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
        for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
    ]
    fastprop_data.to_csv(Path(src_file).stem + "_features.csv")
