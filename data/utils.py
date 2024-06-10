from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem

DESCRIPTOR_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]


def get_descs(src_df: pd.DataFrame):
    """Calculates features for solute and solvent, returns DataFrame for writing to CSV.

    Args:
        src_df (pd.DataFrame): DataFrame with 'solvent_smiles', 'solute_smiles', 'temperature', and 'logS'.
                               Other columns will be ignored.
    """
    unique_smiles: np.ndarray = np.hstack((pd.unique(src_df["solvent_smiles"]), pd.unique(src_df["solute_smiles"])))
    descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)
    # assemble the data into the format expected in fastprop
    # map smiles -> descriptors
    smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
    fastprop_data: pd.DataFrame = src_df[["solute_smiles", "solvent_smiles", "logS", "temperature"]]
    fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + DESCRIPTOR_COLUMNS)
    fastprop_data[DESCRIPTOR_COLUMNS] = [
        np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
        for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
    ]
    return fastprop_data


def drop_bigsol_overlap(src_df: pd.DataFrame):
    """Drops entries with solutes that are in BigSolDB

    Args:
        src_df (pd.DataFrame): Data to be possibly dropped.

    Raises:
        RuntimeError: Missing processed BigSolDB file.

    Returns:
        pd.DataFrame: src_df with overlap dropped and index reset.
    """
    try:
        bigsol_smiles = pl.read_csv(Path("krasnov/bigsoldb_downsample.csv"), columns=["solute_smiles"])["solute_smiles"].to_list()
    except FileNotFoundError as e:
        raise RuntimeError("Unable to open BigSol - run `python krasnov.py` first.") from e
    src_smiles = set(Chem.CanonSmiles(s) for s in src_df["solute_smiles"])
    bigsol_smiles = set(bigsol_smiles)
    overlapped_smiles = tuple(src_smiles.intersection(bigsol_smiles))
    print(len(src_df), "<--- number of entries in the source data")
    src_df = src_df[~src_df["solute_smiles"].isin(overlapped_smiles)].reset_index(drop=True)
    print(len(src_df), "<--- number of entries after dropping solutes in our training dataset")
    return src_df
