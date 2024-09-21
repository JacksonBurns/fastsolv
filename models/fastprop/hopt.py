import logging
import os
from typing import Dict
from pathlib import Path

import pandas as pd
import ray
import torch
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import fastprop
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from train import SOLUTE_COLUMNS, SOLVENT_COLUMNS, train_ensemble

logger = init_logger(__name__)
ray.init(_temp_dir='/state/partition1/user/jburns', num_cpus=40, num_gpus=2)

NUM_HOPT_TRIALS = 128


def define_by_run_func(trial):
    trial.suggest_categorical("input_activation", ("sigmoid", "tanh", "clamp3"))
    trial.suggest_categorical("activation_fxn", ("relu", "leakyrelu", "sigmoid", "tanh"))
    trial.suggest_int("hidden_size", 400, 3_400, 200)
    trial.suggest_int("num_layers", 0, 6, 1)
    trial.suggest_int("solvent_hidden_size", 400, 3_400, 200)
    trial.suggest_int("solvent_layers", 0, 6, 1)
    trial.suggest_int("solute_hidden_size", 400, 3_400, 200)
    trial.suggest_int("solute_layers", 0, 6, 1)


def main():
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # load the data
    df = pd.read_csv(Path("../../data/krasnov/bigsoldb_downsample.csv"), index_col=0)
    metadata_df = df[["solute_smiles", "solvent_smiles", "source"]]
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
    metadata_df_ref = ray.put(metadata_df)
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: _hopt_objective(
                trial,
                solubilites_ref,
                temperatures_ref,
                solute_features_ref,
                solvent_features_ref,
                metadata_df_ref,
            ),
            resources={"gpu": 1, "cpu": 20},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            max_concurrent_trials=2,
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
    metadata_df_ref,
) -> Dict[str, float]:
    solubilities = ray.get(solubilites_ref)
    temperatures = ray.get(temperatures_ref)
    solute_features = ray.get(solute_features_ref)
    solvent_features = ray.get(solvent_features_ref)
    metadata_df = ray.get(metadata_df_ref)
    validation_results_df = train_ensemble(
        data=(solute_features, solvent_features, temperatures, solubilities, metadata_df),
        remove_output=True,
        num_features=1_613,
        learning_rate=0.0001,
        **trial,
    )
    return {"mse": validation_results_df.describe().at["mean", "validation_mse_scaled_loss"]}


if __name__ == "__main__":
    main()
