# llompart.py
#
# Usage: python llompart.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV file:
# https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CZVZIA#
#
# Running this will then csv files for fastprop predicting.
from pathlib import Path

import pandas as pd
from rdkit import Chem

from utils import get_descs

DROP_OVERLAP = False
# ends up removing these six molecules
# {
#     "N#Cc1c(Cl)c(Cl)c(Cl)c(C#N)c1Cl",
#     "CCOC(=O)c1ccc(O)cc1",
#     "O=C(c1ccccc1)C(O)c1ccccc1",
#     "CC(O)(CS(=O)(=O)c1ccc(F)cc1)C(=O)Nc1ccc(C#N)c(C(F)(F)F)c1",
#     "c1ccc2[nH]c(-c3cscn3)nc2c1",
#     "OC1C(O)C(O)C(O)C(O)C1O",
# }

llompart_data: pd.DataFrame = pd.read_csv("OChemUnseen.csv")
if DROP_OVERLAP:
    print(len(llompart_data), "<--- number of molecules in the original dataset")
    # drop any which are also in our data
    vermeire_smiles = set(Chem.CanonSmiles(s) for s in pd.read_csv(Path("vermeire/prepared_data.csv"), index_col=0)["solute_smiles"])
    llompart_smiles = set(Chem.CanonSmiles(s) for s in llompart_data["SMILES"])
    overlapped_smiles = tuple(vermeire_smiles.intersection(llompart_smiles))
    llompart_data = llompart_data[~llompart_data["SMILES"].isin(overlapped_smiles)]
    print(len(llompart_data), "<--- number of molecules after dropping those in our training dataset")

llompart_data.insert(0, "temperature", 273.15 + 25)
llompart_data.insert(0, "solvent_smiles", "O")
llompart_data = llompart_data.rename(columns={"LogS": "logS", "T": "temperature", "SMILES": "solute_smiles"})

fastprop_data = get_descs(llompart_data)

_dest = Path("llompart")
if not Path.exists(_dest):
    Path.mkdir(_dest)
fastprop_data.to_csv(_dest / "llompart_features.csv")
