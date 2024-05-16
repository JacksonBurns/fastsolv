import logging
import os
from typing import Dict
from pathlib import Path

import pandas as pd
import psutil
import ray
import torch
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import fastprop
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from train import SOLUTE_COLUMNS, SOLVENT_COLUMNS, train_ensemble

logger = init_logger(__name__)

NUM_HOPT_TRIALS = 128
ENABLE_BRANCHES = True


def define_by_run_func(trial):
    trial.suggest_categorical("act_fun", ("tanh", "relu", "relu6", "sigmoid", "leakyrelu"))  # , "relun"
    trial.suggest_int("interaction_hidden_size", 400, 3_400, 100)
    trial.suggest_int("num_interaction_layers", 0, 4, 1)
    interaction = trial.suggest_categorical("interaction", ("concatenation", "multiplication", "subtraction"))  # "pairwisemax",

    if ENABLE_BRANCHES:
        # if either solute OR solvent has hidden layers (but NOT both), can only do concatenation or pairwisemax
        if interaction in {"concatenation", "pairwisemax"}:
            trial.suggest_int("solute_layers", 0, 4, 1)
            trial.suggest_int("solute_hidden_size", 200, 2_200, 100)
            trial.suggest_int("solvent_layers", 0, 4, 1)
            trial.suggest_int("solvent_hidden_size", 200, 2_200, 100)
        else:
            solute_layers = trial.suggest_int("solute_layers", 0, 4, 1)
            if solute_layers == 0:
                trial.suggest_int("solvent_layers", 0, 0)
                trial.suggest_int("solute_hidden_size", 0, 0)
                trial.suggest_int("solvent_hidden_size", 0, 0)
            else:
                trial.suggest_int("solvent_layers", 1, 4, 1)
                matched_hidden_size = trial.suggest_int("solute_hidden_size", 200, 2_200, 100)
                trial.suggest_int("solvent_hidden_size", matched_hidden_size, matched_hidden_size)
    else:
        return {"solute_layers": 0, "solute_hidden_size": 0, "solvent_layers": 0, "solvent_hidden_size": 0}


def main():
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # load the data
    df = pd.read_csv(Path("../../data/vermeire/prepared_data.csv"), index_col=0)
    smiles_df = df[["solute_smiles", "solvent_smiles"]]
    solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
    temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
    solvent_features = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)

    logger.info("Run 'tensorboard --logdir output/tensorboard_logs' to track training progress.")
    metric = fastprop.get_metric("regression")
    algo = OptunaSearch(space=define_by_run_func, metric=metric, mode="min")
    solubilites_ref = ray.put(solubilities)
    temperatures_ref = ray.put(temperatures)
    solute_features_ref = ray.put(solute_features)
    solvent_features_ref = ray.put(solvent_features)
    smiles_df_ref = ray.put(smiles_df)
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: _hopt_objective(
                trial,
                solubilites_ref,
                temperatures_ref,
                solute_features_ref,
                solvent_features_ref,
                smiles_df_ref,
            ),
            resources={"gpu": 1, "cpu": psutil.cpu_count()},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            max_concurrent_trials=1,
            num_samples=NUM_HOPT_TRIALS,
            metric=metric,
            mode="min",
        ),
    )
    results = tuner.fit()
    results.get_dataframe().to_csv("hopt_results.csv")
    best = results.get_best_result().config
    logger.info(f"Best hyperparameters identified: {', '.join([key + ': ' + str(val) for key, val in best.items()])}")
    return best


def _hopt_objective(
    trial,
    solubilites_ref,
    temperatures_ref,
    solute_features_ref,
    solvent_features_ref,
    smiles_df_ref,
) -> Dict[str, float]:
    solubilities = ray.get(solubilites_ref)
    temperatures = ray.get(temperatures_ref)
    solute_features = ray.get(solute_features_ref)
    solvent_features = ray.get(solvent_features_ref)
    smiles_df = ray.get(smiles_df_ref)
    validation_results_df, _ = train_ensemble(
        data=(solute_features, solvent_features, temperatures, solubilities, smiles_df),
        remove_output=True,
        run_holdout=False,
        num_solute_layers=trial["solute_layers"],
        solute_hidden_size=trial["solute_hidden_size"],
        num_solvent_layers=trial["solvent_layers"],
        solvent_hidden_size=trial["solvent_hidden_size"],
        num_interaction_layers=trial["num_interaction_layers"],
        interaction_hidden_size=trial["interaction_hidden_size"],
        interaction_operation=trial["interaction"],
        activation_fxn=trial["act_fun"],
        num_features=1_613,
        learning_rate=0.0001,
    )
    return {"mse": validation_results_df.describe().at["mean", "validation_mse_scaled_loss"]}


if __name__ == "__main__":
    main()
