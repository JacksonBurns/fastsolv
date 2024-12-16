"""
solprop_boobier.py

Runs the Vermeire model on the Boobier Datasets as well as the held out set from the
source study.

Install solprop with this command:
conda create -n solprop -c fhvermei python=3.7 solprop_ml
conda install -n solprop -c rmg descriptastorus

Taken from this github comment:
https://github.com/fhvermei/SolProp_ML/issues/6#issuecomment-1317395817

Be sure to then also install polars, either via pip or conda.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from _solprop import make_solubility_prediction
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

for dataset in (
    Path("../data/boobier/leeds_acetone.csv"),
    Path("../data/boobier/leeds_benzene.csv"),
    Path("../data/boobier/leeds_ethanol.csv"),
    Path("../data/vermeire/solprop_nonaq.csv"),
):
    df = pl.read_csv(dataset, columns=["solute_smiles", "solvent_smiles", "temperature", "logS"])
    solute_list = df["solute_smiles"].to_list()
    solvent_list = df["solvent_smiles"].to_list()
    temp_list = df["temperature"].to_list()
    logS_truth = df["logS"].to_list()

    ref_solvent_list = [None] * len(solute_list)
    ref_solubility_list = [None] * len(solute_list)
    ref_temp_list = [None] * len(solute_list)
    hsub298_list = [None] * len(solute_list)
    cp_gas_298_list = [None] * len(solute_list)
    cp_solid_298_list = [None] * len(solute_list)

    df_results = make_solubility_prediction(
        solvent_list=solvent_list,
        solute_list=solute_list,
        temp_list=temp_list,
        ref_solvent_list=ref_solvent_list,
        ref_solubility_list=ref_solubility_list,
        ref_temp_list=ref_temp_list,
        hsub298_list=hsub298_list,
        cp_gas_298_list=cp_gas_298_list,
        cp_solid_298_list=cp_solid_298_list,
    )
    df_results.to_csv(dataset.stem + "_vermeire_predictions.csv")

    preds = df_results["logST (method1) [log10(mol/L)]"].to_list()
    abs_err = np.array([np.abs(i - j) for i, j in zip(preds, logS_truth)])

    r, _ = pearsonr(preds, logS_truth)
    mse = mean_squared_error(preds, logS_truth)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(preds, logS_truth)
    wn_07 = np.count_nonzero(abs_err <= 0.7) / len(preds)
    wn_1 = np.count_nonzero(abs_err <= 1.0) / len(preds)
    stat_str = f" - Pearson's r: {r:.4f}\n - MAE: {mae:.4f}\n - MSE: {mse:.4f}\n - RMSE: {rmse:.4f}\n - W/n 0.7: {wn_07:.4f}\n - W/n 1.0: {wn_1:.4f}"
    plt.clf()
    plt.scatter(logS_truth, preds, alpha=0.1)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    min_val = min(np.min(logS_truth), np.min(preds)) - 0.5
    max_val = max(np.max(logS_truth), np.max(preds)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    plt.plot([min_val, max_val], [min_val + 1, max_val + 1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val - 1, max_val - 1], color="red", linestyle="--", alpha=0.25)
    plt.ylim(min_val, max_val)
    plt.xlim(min_val, max_val)
    plt.text(min_val, max_val - 0.1, stat_str, horizontalalignment="left", verticalalignment="top")
    plt.title(dataset.stem)
    plt.savefig(Path("../analysis/figures", dataset.stem + "_vermeire_results.png"))
