# llompart.py
#
# Usage: python llompart.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the CSV files at:
# https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CZVZIA#
#
# Running this will then csv files for fastprop predicting.
#
# Follows the curation procedure described in:
# https://doi.org/10.1038/s41597-024-03105-6
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from utils import get_descs

_dest = Path("llompart")
if not Path.exists(_dest):
    Path.mkdir(_dest)

disallowed_atoms = {
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
}


def _keep(s):
    if not isinstance(s, str):
        return False
    if "." in s:
        return False
    mol = Chem.MolFromSmiles(s)
    remover = SaltRemover()
    new_mol = remover.StripMol(mol)
    if mol.GetNumAtoms() != new_mol.GetNumAtoms():
        return False
    if mol.GetNumAtoms() < 2:
        return False
    for atom in disallowed_atoms:
        if atom in s:
            return False
    return True


# AqSolDB original
df = pd.read_csv("AqSolDBc_butnotactually.csv")
df.insert(0, "temperature", 273.15 + 25)
df.insert(0, "solvent_smiles", "O")
df = df.rename(columns={"Solubility": "logS", "SMILEScurated": "solute_smiles"})
df = df[~df["solute_smiles"].isna()].reset_index()

fastprop_data = get_descs(df)

fastprop_data.insert(1, "source", "unknown")
fastprop_data.to_csv(_dest / "aqsoldb_og.csv")

exit(1)

# AqSolDBc
df = pd.read_csv("AqSolDBc.csv")
df.insert(0, "temperature", 273.15 + 25)
df.insert(0, "solvent_smiles", "O")
df = df.rename(columns={"ExperimentalLogS": "logS", "SmilesCurated": "solute_smiles"})

fastprop_data = get_descs(df)

fastprop_data.insert(1, "source", "unknown")
fastprop_data.to_csv(_dest / "llompart_aqsoldbc.csv")

# OChemUnseen (starts from Curated to apply the same preprocessing)
df: pd.DataFrame = pd.read_csv("OChemCurated.csv")
print(len(df), "<-- original OChemCurated")
smiles_to_temp = {smi: temp for smi, temp in zip(df["SMILES"], df["Temperature"], strict=True)}
to_drop = df["SMILES"][~(
    df["SMILES"].apply(_keep)  # no missing SMILES, no salts, 2+ atoms, no banned atoms
    & df["SDi"].le(0.5))  # deviation less than 0.5
].to_list()
print(len(to_drop), "<-- eligible to be dropped")

df: pd.DataFrame = pd.read_csv("OChemUnseen.csv")
print(len(df), "<-- non-overlapping")
df = df[~df["SMILES"].isin(to_drop)]
df.insert(0, "solvent_smiles", "O")
df.insert(0, "temperature", [smiles_to_temp.get(s, 25) + 273.15 for s in df["SMILES"]])
df = df.rename(columns={"SMILES": "solute_smiles", "LogS": "logS", "Temperature": "temperature"}).reset_index()
print(len(df), "<-- non-overlapping, dropped")

fastprop_data = get_descs(df)

fastprop_data.to_csv(_dest / "llompart_ochem.csv")
