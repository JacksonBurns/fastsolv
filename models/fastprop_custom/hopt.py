import logging
import os
from typing import Dict

import pandas as pd
import psutil
import ray
import torch
from astartes import train_test_split
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import fastprop, train_and_test
from lightning.pytorch import seed_everything
from ray import tune
from ray.train.torch import enable_reproducibility
from ray.tune.search.optuna import OptunaSearch

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)

NUM_HOPT_TRIALS = 1
ENABLE_BRANCHES = True
NUM_REPLICATES = 1


# TODO: nested repetitions (change test set)

def define_by_run_func(trial):
    trial.suggest_int("interaction_hidden_size", 400, 3_200, 400)
    trial.suggest_int("branch_hidden_size", 200, 2_200, 400)
    trial.suggest_int("interaction_layers", 0, 3, 1)
    interaction = trial.suggest_categorical("interaction", ("concatenation", "multiplication", "subtraction"))

    if ENABLE_BRANCHES:
        # if either solute OR solvent has hidden layers (but NOT both), can only do concatenation
        solvent_layers = trial.suggest_int("solvent_layers", 0, 3, 1)
        if interaction == "concatenation":
            trial.suggest_int("solute_layers", 0, 3, 1)
        else:
            if solvent_layers == 0:
                trial.suggest_int("solute_layers", 0, 0)
            else:
                trial.suggest_int("solute_layers", 1, 3, 1)
    else:
        return {"solute_layers": 0, "solvent_layers": 0}


def main():
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(42)

    # load the data
    df = pd.read_csv("prepared_data.csv", index_col=0)
    solute_df = df[["solute_smiles"]]
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
    solute_df_ref = ray.put(solute_df)
    tuner = tune.Tuner(
        tune.with_resources(
            lambda trial: _hopt_objective(
                trial,
                solubilites_ref,
                temperatures_ref,
                solute_features_ref,
                solvent_features_ref,
                solute_df_ref,
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
    solute_df_ref,
) -> Dict[str, float]:
    solubilities = ray.get(solubilites_ref)
    temperatures = ray.get(temperatures_ref)
    solute_features = ray.get(solute_features_ref)
    solvent_features = ray.get(solvent_features_ref)
    solute_df = ray.get(solute_df_ref)
    random_seed = 42
    enable_reproducibility(random_seed)
    solutes_hopt, solutes_test = train_test_split(pd.unique(solute_df["solute_smiles"]), random_state=random_seed)
    test_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_test)].tolist()
    hopt_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_hopt)].tolist()
    all_test_results, all_validation_results = [], []
    for replicate_number in range(NUM_REPLICATES):
        logger.info(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")

        # split the data s.t. some solutes are not seen during training
        solutes_train, solutes_val = train_test_split(pd.unique(solute_df["solute_smiles"][hopt_indexes]), random_state=random_seed)
        train_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_train)].tolist()
        val_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_val)].tolist()

        # rescale ALL the things
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
            trial["branch_hidden_size"],
            trial["interaction_layers"],
            trial["interaction_hidden_size"],
            trial["interaction"],
            1_613,
            0.001,
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
    return {"mse": validation_results_df.describe().at["mean", "validation_mse_scaled_loss"]}


if __name__ == "__main__":
    main()
