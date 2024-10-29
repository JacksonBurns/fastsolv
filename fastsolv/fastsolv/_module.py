import os
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from fastprop.data import fastpropDataLoader
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from pytorch_lightning import Trainer
from rdkit import Chem
from tqdm import tqdm

from ._classes import SolubilityDataset, _fastsolv

_SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
_SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]
_DESCRIPTOR_COLUMNS = _SOLUTE_COLUMNS + _SOLVENT_COLUMNS


_ALL_MODELS = []
ckpt_dir = Path(__file__).parent / "checkpoints"
if not ckpt_dir.exists():
    try:
        ckpt_dir.mkdir()
        for i in tqdm(range(1, 5), desc="Downloading model files from Zenodo"):
            urlretrieve(rf"https://zenodo.org/records/13943074/files/fastsolv_1701_{i}.ckpt", ckpt_dir / f"fastsolv_1701_{i}.ckpt")
    except Exception as e:
        raise RuntimeError(
            f"Unable to download model files - try re-running or manually download the checkpoints from zenodo.org/records/13943074 into {ckpt_dir}."
        ) from e
for checkpoint in os.listdir(ckpt_dir):
    model = _fastsolv.load_from_checkpoint(os.path.join(ckpt_dir, checkpoint))
    _ALL_MODELS.append(model)


def fastsolv(df: pd.DataFrame) -> pd.DataFrame:
    """fastsolv solubility predictor

    Args:
        df (pd.DataFrame): DataFrame with 'solute_smiles', 'solvent_smiles', and 'temperature' columns.

    Returns:
        pd.DataFrame: Predicted logS and stdev.
    """
    # calculate the descriptors
    unique_smiles: np.ndarray = np.hstack((pd.unique(df["solvent_smiles"]), pd.unique(df["solute_smiles"])))
    descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)
    # assemble the data into the format expected in fastprop
    # map smiles -> descriptors
    smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
    fastprop_data: pd.DataFrame = df[["solute_smiles", "solvent_smiles", "temperature"]]
    fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + _DESCRIPTOR_COLUMNS)
    fastprop_data[_DESCRIPTOR_COLUMNS] = [
        np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
        for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
    ]
    descs = torch.tensor(descs, dtype=torch.float32)
    predict_dataloader = fastpropDataLoader(
        SolubilityDataset(
            torch.tensor(fastprop_data[_SOLUTE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32),
            torch.tensor(fastprop_data[_SOLVENT_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32),
            torch.tensor(fastprop_data["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(-1),
            torch.zeros(len(df), dtype=torch.float32),
            torch.zeros(len(df), dtype=torch.float32),
        ),
        num_workers=0,
        persistent_workers=False,
    )
    # run inference
    # axis: contents
    # 0: smiles
    # 1: predictions
    # 2: per-model
    trainer = Trainer(logger=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*does not have many workers.*")
        all_predictions = np.stack([torch.vstack(trainer.predict(model, predict_dataloader)).numpy(force=True) for model in _ALL_MODELS], axis=2)
    perf = np.mean(all_predictions, axis=2)
    err = np.std(all_predictions, axis=2)
    # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
    res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
    res[:, 0::2] = perf
    res[:, 1::2] = err
    column_names = ["predicted_logS", "predicted_logS_stdev"]
    return pd.DataFrame(res, columns=column_names, index=pd.MultiIndex.from_frame(df[["solute_smiles", "solvent_smiles", "temperature"]]))
