import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastprop.data import fastpropDataLoader
from fastprop.defaults import ALL_2D
from pytorch_lightning import Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from data import SolubilityDataset
from model import fastpropSolubility

SCALE_TARGETS = True
SOLUTE_EXTRAPOLATION = True
RANDOM_SEED = 1701  # the final frontier

SOLUTE_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS: list[str] = ["solvent_" + d for d in ALL_2D]


def parity_plot(truth, prediction, title, out_fpath):
    plt.scatter(truth, prediction, alpha=0.1)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    min_val = min(np.min(truth), np.min(prediction)) - 0.5
    max_val = max(np.max(truth), np.max(prediction)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    plt.plot([min_val, max_val], [min_val+1, max_val+1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val-1, max_val-1], color="red", linestyle="--", alpha=0.25)
    plt.ylim(min_val, max_val)
    plt.xlim(min_val, max_val)
    plt.title(title)
    plt.savefig(out_fpath)
    plt.show()


def test_ensemble(checkpoint_dir: Path):
    _output_dir = checkpoint_dir.parents[0]
    # reload the models as an ensemble
    all_models = []
    for checkpoint in os.listdir(checkpoint_dir):
        model = fastpropSolubility.load_from_checkpoint(checkpoint_dir / checkpoint)
        all_models.append(model)
    for holdout_fpath, holdout_name in zip(
        (
            Path("boobier/acetone_solubility_data_features.csv"),
            Path("boobier/benzene_solubility_data_features.csv"),
            Path("boobier/ethanol_solubility_data_features.csv"),
            Path("llompart/llompart_features.csv"),
            Path("krasnov/bigsol_features.csv"),
            Path("vermeire/prepared_data.csv"),
        ),
        (
            "boobier_acetone",
            "boobier_benzene",
            "boobier_ethanol",
            "llompart",
            "krasnov",
            "vermeire",
        ),
    ):
        # load the holdout data
        df = pd.read_csv(Path("../../data") / holdout_fpath, index_col=0)
        solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        solvent_features = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)
        smiles = df[["solvent_smiles", "solute_smiles"]].apply(lambda row: ".".join(row), axis=1).tolist()
        dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features,
                solvent_features,
                temperatures,
                solubilities,
            ),
            batch_size=len(solubilities),
        )
        # run inference
        # axis: contents
        # 0: smiles
        # 1: predictions
        # 2: per-model
        trainer = Trainer(logger=False, enable_progress_bar=False)
        all_predictions = np.stack([torch.vstack(trainer.predict(model, dataloader)).numpy(force=True) for model in all_models], axis=2)
        perf = np.mean(all_predictions, axis=2)
        err = np.std(all_predictions, axis=2)
        # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
        res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
        res[:, 0::2] = perf
        res[:, 1::2] = err
        out = pd.DataFrame(res, columns=["logS_pred", "stdev"], index=smiles)
        out.index.name = 'smiles'
        out.insert(0, "logS_true", df["logS"].tolist())
        out.to_csv(_output_dir / (holdout_fpath.stem + "_predictions.csv"))

        # performance metrics
        r, _ = pearsonr(out['logS_true'], out['logS_pred'])
        mse = mean_squared_error(out['logS_true'], out['logS_pred'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(out['logS_true'], out['logS_pred'])
        print(f"Metrics for {holdout_name}:\n - Pearson's r: {r:.4f}\n - MAE: {mae:.4f}\n - MSE: {mse:.4f}\n - RMSE: {rmse:.4f}")
        parity_plot(out['logS_true'], out['logS_pred'], holdout_name, _output_dir / f"{holdout_name}_parity.png")


if __name__ == "__main__":
    test_ensemble(Path("output/fastprop_1715737952/checkpoints"))
