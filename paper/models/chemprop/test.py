import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chemprop import data as chemprop_data_utils
from chemprop import featurizers
from chemprop.models import load_model
from lightning.pytorch import Trainer
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train import (
    CustomMSEMetric,  # needed to load model (or do we just need to call the registry?)
)


def parity_plot(truth, prediction, title, out_fpath, stat_str):
    plt.clf()
    plt.scatter(truth, prediction, alpha=0.1)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    min_val = min(np.min(truth), np.min(prediction)) - 0.5
    max_val = max(np.max(truth), np.max(prediction)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    plt.plot([min_val, max_val], [min_val + 1, max_val + 1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val - 1, max_val - 1], color="red", linestyle="--", alpha=0.25)
    plt.ylim(min_val, max_val)
    plt.xlim(min_val, max_val)
    plt.text(min_val, max_val - 0.1, stat_str, horizontalalignment="left", verticalalignment="top")
    plt.title(title)
    plt.savefig(out_fpath)
    print("wrote plot to", out_fpath)
    # plt.show()


def test_ensemble(checkpoint_dir: Path):
    _output_dir = checkpoint_dir.parents[0]
    # reload the models as an ensemble
    all_models = []
    for checkpoint in os.listdir(checkpoint_dir):
        model = load_model(checkpoint_dir / checkpoint, multicomponent=True)
        all_models.append(model)
    rmses = []
    for holdout_fpath in (
        Path("boobier/leeds_acetone_chemprop.csv"),
        Path("boobier/leeds_benzene_chemprop.csv"),
        Path("boobier/leeds_ethanol_chemprop.csv"),
        Path("vermeire/solprop_chemprop_nonaq.csv"),
    ):
        # load the holdout data
        df = pd.read_csv(Path("../../data") / holdout_fpath, index_col=0)
        test_datapoints = [
            [
                chemprop_data_utils.MoleculeDatapoint.from_smi(smi, None, x_d=np.array([temperature]))
                for smi, temperature in zip(df["solute_smiles"], df["temperature"])
            ],
            list(map(chemprop_data_utils.MoleculeDatapoint.from_smi, df["solvent_smiles"])),
        ]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_datasets = [chemprop_data_utils.MoleculeDataset(test_datapoints[i], featurizer) for i in range(len(test_datapoints))]
        test_mcdset = chemprop_data_utils.MulticomponentDataset(test_datasets)
        test_loader = chemprop_data_utils.build_dataloader(test_mcdset, shuffle=False)
        # run inference
        # axis: contents
        # 0: smiles
        # 1: predictions
        # 2: per-model
        trainer = Trainer(logger=False, enable_progress_bar=False)
        all_predictions = np.stack([torch.vstack(trainer.predict(model, test_loader)).numpy(force=True) for model in all_models], axis=2)
        perf = np.mean(all_predictions, axis=2)
        err = np.std(all_predictions, axis=2)
        # interleave the columns of these arrays, thanks stackoverflow.com/a/75519265
        res = np.empty((len(perf), perf.shape[1] * 2), dtype=perf.dtype)
        res[:, 0::2] = perf
        res[:, 1::2] = err
        out = pd.DataFrame(res, columns=["logS_pred", "stdev"], index=np.arange(len(df)))
        out.insert(0, "solute_smiles", df["solute_smiles"].tolist())
        out.insert(1, "solvent_smiles", df["solvent_smiles"].tolist())
        out.insert(2, "logS_true", df["logS"].tolist())
        out.insert(3, "temperature", df["temperature"].tolist())
        out.to_csv(_output_dir / (holdout_fpath.stem + "_predictions.csv"))

        # performance metrics
        r, _ = pearsonr(out["logS_true"], out["logS_pred"])
        mse = mean_squared_error(out["logS_true"], out["logS_pred"])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(out["logS_true"], out["logS_pred"])
        wn_07 = np.count_nonzero(np.abs(out["logS_true"] - out["logS_pred"]) <= 0.7) / len(out["logS_pred"])
        wn_1 = np.count_nonzero(np.abs(out["logS_true"] - out["logS_pred"]) <= 1.0) / len(out["logS_pred"])
        stat_str = (
            f" - Pearson's r: {r:.4f}\n - MAE: {mae:.4f}\n - MSE: {mse:.4f}\n - RMSE: {rmse:.4f}\n - W/n 0.7: {wn_07:.4f}\n - W/n 1.0: {wn_1:.4f}"
        )
        parity_plot(out["logS_true"], out["logS_pred"], holdout_fpath.stem, _output_dir / f"{holdout_fpath.stem}_parity.png", stat_str)
        rmses.append(rmse)
    return rmses


if __name__ == "__main__":
    test_ensemble(Path("output/idek/checkpoints"))
    exit(0)
    # chemprop_sobolev_leeds_results = []
    # chemprop_sobolev_solprop_results = []
    # for training_count in (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215):
    #     leeds_acetone, leeds_benzene, leeds_ethanol, solprop = test_ensemble(Path(f"output/chemprop_{training_count}/checkpoints"))
    #     chemprop_sobolev_leeds_results.append([leeds_acetone, leeds_benzene, leeds_ethanol])
    #     chemprop_sobolev_solprop_results.append(solprop)
    # print(f"{chemprop_sobolev_solprop_results=}")
    # print(f"{chemprop_sobolev_leeds_results=}")
