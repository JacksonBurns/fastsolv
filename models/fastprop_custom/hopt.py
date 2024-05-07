import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything

from fastprop.data import (
    fastpropDataLoader,
    split,
    standard_scale,
)
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import fastprop, train_and_test

import ray
from ray import tune
from ray.train.torch import enable_reproducibility
from ray.tune.search.optuna import OptunaSearch

from model import fastpropSolubility
from data import SolubilityDataset

logger = init_logger(__name__)

NUM_HOPT_TRIALS = 1


def define_by_run_func(trial):
    trial.suggest_categorical("hidden_size", tuple(range(400, 1901, 500)))
    trial.suggest_categorical("interaction_layers", tuple(range(0, 3, 1)))
    solvent_layers = trial.suggest_categorical("solvent_layers", tuple(range(1, 3, 1)))
    solute_layers = trial.suggest_categorical("solute_layers", tuple(range(0, 1, 1)))

    # if either solute OR solvent has hidden layers (but NOT both), can only do concatenation
    if (solvent_layers == 0 and solute_layers > 0) or (solvent_layers > 0 and solute_layers == 0):
        trial.suggest_categorical("interaction", ("concatenation",))
    else:
        trial.suggest_categorical("interaction", ("multiplication", "subtraction"))


def main():
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("output", "checkpoints"), exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(42)

    # load the data
    df = pd.read_csv("prepared_data.csv", index_col=0)
    solubilities = torch.tensor(df.iloc[:, 2].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
    temperatures = torch.tensor(df.iloc[:, 3].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    solute_features = torch.tensor(df.iloc[:, 4 : (4 + 1_613)].to_numpy(), dtype=torch.float32)
    solvent_features = torch.tensor(df.iloc[:, (4 + 1_613) :].to_numpy(), dtype=torch.float32)

    logger.info("Run 'tensorboard --logdir output/tensorboard_logs' to track training progress.")
    metric = fastprop.get_metric("regression")
    algo = OptunaSearch(space=define_by_run_func, metric=metric, mode="min")
    solubilites_ref = ray.put(solubilities)
    temperatures_ref = ray.put(temperatures)
    solute_features_ref = ray.put(solute_features)
    solvent_features_ref = ray.put(solvent_features)
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: _hopt_objective(
                trial,
                solubilites_ref,
                temperatures_ref,
                solute_features_ref,
                solvent_features_ref,
            ),
            # run 2 models at the same time (leave 20% for system)
            # don't specify cpus, and just let pl figure it out
            resources={"gpu": (1 - 0.20) / 2},
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            max_concurrent_trials=2,
            num_samples=NUM_HOPT_TRIALS,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result().config
    logger.info(f"Best hyperparameters identified: {', '.join([key + ': ' + str(val) for key, val in best.items()])}")
    return best


def _hopt_objective(
    trial,
    solubilites_ref,
    temperatures_ref,
    solute_features_ref,
    solvent_features_ref,
) -> Dict[str, float]:
    solubilities = ray.get(solubilites_ref)
    temperatures = ray.get(temperatures_ref)
    solute_features = ray.get(solute_features_ref)
    solvent_features = ray.get(solvent_features_ref)
    random_seed = 42
    enable_reproducibility(random_seed)
    all_test_results, all_validation_results = [], []
    for replicate_number in range(3):
        logger.info(f"Training model {replicate_number+1} of 3 ({random_seed=})")

        # rescale ALL the things
        train_indexes, val_indexes, test_indexes = split(np.arange(len(solubilities)))
        solute_features[train_indexes], solute_feature_means, solute_feature_vars = standard_scale(solute_features[train_indexes])
        solute_features[val_indexes] = standard_scale(solute_features[val_indexes], solute_feature_means, solute_feature_vars)
        solute_features[test_indexes] = standard_scale(solute_features[test_indexes], solute_feature_means, solute_feature_vars)

        solvent_features[train_indexes], solvent_feature_means, solvent_feature_vars = standard_scale(solvent_features[train_indexes])
        solvent_features[val_indexes] = standard_scale(solvent_features[val_indexes], solvent_feature_means, solvent_feature_vars)
        solvent_features[test_indexes] = standard_scale(solvent_features[test_indexes], solvent_feature_means, solvent_feature_vars)

        solubilities[train_indexes], solubility_means, solubility_vars = standard_scale(solubilities[train_indexes])
        solubilities[val_indexes] = standard_scale(solubilities[val_indexes], solubility_means, solubility_vars)
        solubilities[test_indexes] = standard_scale(solubilities[test_indexes], solubility_means, solubility_vars)

        temperatures[train_indexes], temperature_means, temperature_vars = standard_scale(temperatures[train_indexes])
        temperatures[val_indexes] = standard_scale(temperatures[val_indexes], temperature_means, temperature_vars)
        temperatures[test_indexes] = standard_scale(temperatures[test_indexes], temperature_means, temperature_vars)

        train_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[train_indexes],
                solvent_features[train_indexes],
                temperatures[train_indexes],
                solubilities[train_indexes],
            ),
            shuffle=True,
        )
        val_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[val_indexes],
                solvent_features[val_indexes],
                temperatures[val_indexes],
                solubilities[val_indexes],
            ),
        )
        test_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[test_indexes],
                solvent_features[test_indexes],
                temperatures[test_indexes],
                solubilities[test_indexes],
            ),
        )

        # initialize the model and train/test
        model = fastpropSolubility(
            trial["solute_layers"],
            trial["solvent_layers"],
            trial["interaction"],
            1_613,
            trial["interaction_layers"],
            0.001,
            trial["hidden_size"],
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test(
            "output",
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            50,
            10,
        )
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())
    return {"mse": test_results_df.describe().at["mean", "test_mse_scaled_loss"]}


if __name__ == "__main__":
    main()
