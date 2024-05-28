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

NUM_HOPT_TRIALS = 1024


def define_by_run_func(trial):
    trial.suggest_categorical("activation_fxn", ("relu", "leakyrelu"))
    trial.suggest_categorical("input_activation", ("tanh", "sigmoid"))
    trial.suggest_int("interaction_hidden_size", 400, 2_000, step=100)
    trial.suggest_int("branch_hidden_size", 400, 2_000, step=100)
    trial.suggest_int("num_interaction_layers", 1, 4, step=1)
    trial.suggest_int("num_solute_layers", 1, 4, step=1)
    trial.suggest_int("num_solvent_layers", 1, 4, step=1)
    trial.suggest_int("num_water_layers", 1, 4, step=1)
    return


def main():
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # load the data
    df = pd.read_csv(Path("../../data/vermeire/prepared_data.csv"), index_col=0)
    smiles_df = df[["solute_smiles", "solvent_smiles"]]
    source_df = df[['source']]
    solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
    temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    is_water = torch.tensor(df["is_water"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
    solvent_features = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)

    metric = fastprop.get_metric("regression")
    algo = OptunaSearch(space=define_by_run_func, metric=metric, mode="min")
    solubilites_ref = ray.put(solubilities)
    temperatures_ref = ray.put(temperatures)
    solute_features_ref = ray.put(solute_features)
    solvent_features_ref = ray.put(solvent_features)
    smiles_df_ref = ray.put(smiles_df)
    is_water_ref = ray.put(is_water)
    source_df_ref = ray.put(source_df)
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: _hopt_objective(
                trial,
                solubilites_ref,
                temperatures_ref,
                solute_features_ref,
                solvent_features_ref,
                smiles_df_ref,
                is_water_ref,
                source_df_ref,
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
    results.get_dataframe().to_csv("aqsep_hopt_results.csv")
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
    is_water_ref,
    source_df_ref,
) -> Dict[str, float]:
    solubilities = ray.get(solubilites_ref)
    temperatures = ray.get(temperatures_ref)
    solute_features = ray.get(solute_features_ref)
    solvent_features = ray.get(solvent_features_ref)
    smiles_df = ray.get(smiles_df_ref)
    is_water = ray.get(is_water_ref)
    source_df = ray.get(source_df_ref)
    validation_results_df, _ = train_ensemble(
        data=(solute_features, solvent_features, temperatures, solubilities, smiles_df, is_water, source_df),
        remove_output=True,
        num_features=1_613,
        learning_rate=0.00001,
        **trial,
    )
    return {"mse": validation_results_df.describe().at["mean", "validation_mse_scaled_loss"]}


if __name__ == "__main__":
    main()
