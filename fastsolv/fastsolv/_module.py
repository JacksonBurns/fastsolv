import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastprop.data import fastpropDataLoader
from fastprop.defaults import ALL_2D
from fastprop.descriptors import get_descriptors
from pytorch_lightning import Trainer
from rdkit import Chem

from ._classes import SolubilityDataset, _fastsolv

SOLUTE_COLUMNS = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS = ["solvent_" + d for d in ALL_2D]
DESCRIPTOR_COLUMNS: list[str] = SOLUTE_COLUMNS + SOLVENT_COLUMNS


def fastsolv(df):
    # calculate the descriptors
    unique_smiles: np.ndarray = np.hstack((pd.unique(df["solvent_smiles"]), pd.unique(df["solute_smiles"])))
    descs: np.ndarray = get_descriptors(False, ALL_2D, list(Chem.MolFromSmiles(i) for i in unique_smiles)).to_numpy(dtype=np.float32)
    # assemble the data into the format expected in fastprop
    # map smiles -> descriptors
    smiles_to_descs: dict = {smiles: desc for smiles, desc in zip(unique_smiles, descs)}
    fastprop_data: pd.DataFrame = df[["solute_smiles", "solvent_smiles", "temperature"]]
    fastprop_data: pd.DataFrame = fastprop_data.reindex(columns=fastprop_data.columns.tolist() + DESCRIPTOR_COLUMNS)
    fastprop_data[DESCRIPTOR_COLUMNS] = [
        np.hstack((smiles_to_descs[solute], smiles_to_descs[solvent]))
        for solute, solvent in zip(fastprop_data["solute_smiles"], fastprop_data["solvent_smiles"])
    ]

    # load the models
    all_models = []
    ckpt_dir = Path(__file__).parent / "checkpoints"
    if not ckpt_dir.exists():
        print(
            f"""
This is a pre-release of fastsolv and does not yet support automatic downloading of model weights, since they are not yet released.
Please manually download the trained model files into {ckpt_dir}
"""
        )
        exit(1)
    for checkpoint in os.listdir(ckpt_dir):
        model = _fastsolv.load_from_checkpoint(os.path.join(ckpt_dir, checkpoint))
        all_models.append(model)

    descs = torch.tensor(descs, dtype=torch.float32)
    predict_dataloader = fastpropDataLoader(
        SolubilityDataset(
            torch.tensor(fastprop_data[SOLUTE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32),
            torch.tensor(fastprop_data[SOLVENT_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32),
            torch.tensor(fastprop_data["temperature"].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(-1),
            torch.zeros(len(df), dtype=torch.float32),
            torch.zeros(len(df), dtype=torch.float32),
        ),
    )
    # run inference
    # axis: contents
    # 0: smiles
    # 1: predictions
    # 2: per-model
    trainer = Trainer(logger=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*does not have many workers.*")
        all_predictions = np.stack([torch.vstack(trainer.predict(model, predict_dataloader)).numpy(force=True) for model in all_models], axis=2)
    perf = np.mean(all_predictions, axis=2)
    err = np.std(all_predictions, axis=2)
    # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
    res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
    res[:, 0::2] = perf
    res[:, 1::2] = err
    column_names = ["predicted_logS", "predicted_logS_stdev"]
    return pd.DataFrame(res, columns=column_names, index=pd.MultiIndex.from_frame(df[["solute_smiles", "solvent_smiles", "temperature"]]))
