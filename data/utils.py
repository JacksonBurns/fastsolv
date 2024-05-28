import numpy as np
import pandas as pd
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
    # insert column indicating if something is in water or not
    fastprop_data.insert(4, 'is_water', fastprop_data['solvent_smiles'].apply(lambda s: int(s in {'O', '[OH2]', '[H]O[H]'})))
    # add the descriptors
    fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + DESCRIPTOR_COLUMNS)
    fastprop_data[DESCRIPTOR_COLUMNS] = [
        np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
        for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
    ]
    return fastprop_data
