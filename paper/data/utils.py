from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from rdkit import Chem

DESCRIPTOR_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D] + ["solvent_" + d for d in ALL_2D]
DROP_WATER = True


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


def drop_overlap(src_df: pd.DataFrame, base_set: str):
    """Drops entries with solutes that are in the base_set

    Args:
        src_df (pd.DataFrame): Data to be possibly dropped.

    Raises:
        RuntimeError: Missing processed file of other database.

    Returns:
        pd.DataFrame: src_df with overlap dropped and index reset.
    """
    if base_set == "krasnov":
        try:
            ref_smiles = pl.read_csv(Path(f"krasnov/bigsoldb_chemprop{'_nonaq' if DROP_WATER else ''}.csv"), columns=["solute_smiles"])[
                "solute_smiles"
            ].to_list()
        except FileNotFoundError as e:
            raise RuntimeError("Unable to open BigSol - run `python krasnov.py` first.") from e
    elif base_set == "vermeire":
        try:
            ref_smiles = pl.read_csv(Path(f"vermeire/solprop_chemprop{'_nonaq' if DROP_WATER else ''}.csv"), columns=["solute_smiles"])[
                "solute_smiles"
            ].to_list()
        except FileNotFoundError as e:
            raise RuntimeError("Unable to open SolProp - run `python vermeire.py` first.") from e
    src_canon_smiles = src_df["solute_smiles"].apply(lambda s: Chem.CanonSmiles(s))
    ref_canon_smiles = set(Chem.CanonSmiles(s) for s in ref_smiles)
    print(len(src_df), "<--- number of entries in the source data")
    src_df = src_df[~src_canon_smiles.isin(ref_canon_smiles)].reset_index(drop=True)
    print(len(src_df), f"<--- number of entries after dropping solutes in {base_set} dataset")
    return src_df
