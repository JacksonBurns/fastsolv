# krasnov.py
#
# Usage: python krasnov.py
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
from rdkit import Chem

from utils import get_descs


DROP_OVERLAP = False

bigsol_data: pd.DataFrame = pd.read_csv("BigSolDB.csv")
print(len(bigsol_data), "<--- number of molecules in the original dataset")
bigsol_data = bigsol_data[~bigsol_data["Solvent"].isin(("PEG-400", "PEG-300", "PEG-200"))]
print(len(bigsol_data), "<--- number of molecules without PEG")
# drop any which are also in our data
if DROP_OVERLAP:
    vermeire_smiles = set(Chem.CanonSmiles(s) for s in pd.read_csv(Path("vermeire/prepared_data.csv"), index_col=0)["solute_smiles"])
    bigsol_smiles = set(Chem.CanonSmiles(s) for s in bigsol_data["SMILES"])
    overlapped_smiles = tuple(vermeire_smiles.intersection(bigsol_smiles))
    bigsol_data = bigsol_data[~bigsol_data["SMILES"].isin(overlapped_smiles)]
    print(len(bigsol_data), "<--- number of molecules after dropping solutes in our training dataset")
bigsol_data = bigsol_data[["SMILES", "SMILES_Solvent", "Solubility", "T,K"]].rename(
    columns={"Solubility": "logS", "T,K": "temperature", "SMILES": "solute_smiles", "SMILES_Solvent": "solvent_smiles"}
)
bigsol_data["logS"] = bigsol_data["logS"].apply(np.log10)

fastprop_data = get_descs(bigsol_data)

_dest = Path("krasnov")
if not Path.exists(_dest):
    Path.mkdir(_dest)
fastprop_data.to_csv(_dest / "bigsol_features.csv")
