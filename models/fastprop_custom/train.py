import datetime
import logging
import os
import shutil
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astartes import train_val_test_split
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import ALL_2D, _init_loggers, init_logger
from fastprop.metrics import SCORE_LOOKUP, mean_absolute_error_score, root_mean_squared_error_loss, weighted_mean_absolute_percentage_error_score, r2_score
from fastprop.model import train_and_test
from lightning.pytorch import seed_everything

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)

NUM_REPLICATES = 4
SCALE_TARGETS = True
SOLUTE_EXTRAPOLATION = True
RANDOM_SEED = 1701  # the final frontier
TRAINING_FPATH = Path("krasnov/bigsol_features.csv")
# one of:
# Path("boobier/acetone_solubility_data_features.csv"),
# Path("boobier/benzene_solubility_data_features.csv"),
# Path("boobier/ethanol_solubility_data_features.csv"),
# Path("llompart/llompart_features.csv"),
# Path("krasnov/bigsol_features.csv"),
# Path("vermeire/prepared_data.csv"),

SOLUTE_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS: list[str] = ["solvent_" + d for d in ALL_2D]


def logS_within_0_7_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(0.7).count_nonzero() / prediction.size(dim=0)


def logS_within_1_0_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(1.0).count_nonzero() / prediction.size(dim=0)


def parity_plot(truth, prediction, title, out_fpath):
    plt.scatter(truth, prediction, alpha=0.1)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    min_val = min(np.min(truth), np.min(prediction))
    max_val = max(np.max(truth), np.max(prediction))
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    plt.plot([min_val, max_val], [min_val+1, max_val+1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val-1, max_val-1], color="red", linestyle="--", alpha=0.25)
    plt.ylim(min_val - 1, max_val + 1)
    plt.xlim(min_val - 1, max_val + 1)
    plt.title(title)
    plt.savefig(out_fpath)
    plt.show()


SCORE_LOOKUP["regression"] = (
    logS_within_0_7_percentage,
    logS_within_1_0_percentage,
    root_mean_squared_error_loss,
    mean_absolute_error_score,
    weighted_mean_absolute_percentage_error_score,
    r2_score,
)


def train_ensemble(data=None, remove_output=False, run_holdout=False, **model_kwargs):
    # setup logging and output directories
    _output_dir = Path(f"output/fastprop_{int(datetime.datetime.now(datetime.UTC).timestamp())}")
    os.makedirs(_output_dir, exist_ok=True)
    _init_loggers(_output_dir)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(RANDOM_SEED)
    _data_dir = Path("../../data")
    # load the training data
    if data is None:
        df = pd.read_csv(_data_dir / TRAINING_FPATH, index_col=0)
        smiles_df = df[["solute_smiles", "solvent_smiles"]]
        solubilities_og = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
        temperatures_og = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        solute_features_og = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        solvent_features_og = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)
    else:
        solute_features_og, solvent_features_og, temperatures_og, solubilities_og, smiles_df = data

    logger.info(f"Run 'tensorboard --logdir {_output_dir}/tensorboard_logs' to track training progress.")
    random_seed = RANDOM_SEED
    all_test_results, all_validation_results = [], []
    for replicate_number in range(NUM_REPLICATES):
        logger.info(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")
        # keep backups so repeat trials don't rescale already scaled data
        solubilities = solubilities_og.detach().clone()
        temperatures = temperatures_og.detach().clone()
        solute_features = solute_features_og.detach().clone()
        solvent_features = solvent_features_og.detach().clone()

        # split the data s.t. model only sees a subset of solutes and solvents
        if SOLUTE_EXTRAPOLATION:
            solutes_train, solutes_val, solutes_test = train_val_test_split(pd.unique(smiles_df["solute_smiles"]), random_state=random_seed)
            train_indexes = smiles_df.index[smiles_df["solute_smiles"].isin(solutes_train)].tolist()
            val_indexes = smiles_df.index[smiles_df["solute_smiles"].isin(solutes_val)].tolist()
            test_indexes = smiles_df.index[smiles_df["solute_smiles"].isin(solutes_test)].tolist()
        else:
            train_indexes, val_indexes, test_indexes = train_val_test_split(np.arange(len(smiles_df)), random_state=random_seed)

        # scaling
        solute_features[train_indexes], solute_feature_means, solute_feature_vars = standard_scale(solute_features[train_indexes])
        solute_scaler = partial(standard_scale, means=solute_feature_means, variances=solute_feature_vars)
        solute_features[val_indexes] = solute_scaler(solute_features[val_indexes])
        solute_features[test_indexes] = solute_scaler(solute_features[test_indexes])

        solvent_features[train_indexes], solvent_feature_means, solvent_feature_vars = standard_scale(solvent_features[train_indexes])
        solvent_scaler = partial(standard_scale, means=solvent_feature_means, variances=solute_feature_vars)
        solvent_features[val_indexes] = solvent_scaler(solvent_features[val_indexes])
        solvent_features[test_indexes] = solvent_scaler(solvent_features[test_indexes])

        temperatures[train_indexes], temperature_means, temperature_vars = standard_scale(temperatures[train_indexes])
        temperature_scaler = partial(standard_scale, means=temperature_means, variances=temperature_vars)
        temperatures[val_indexes] = temperature_scaler(temperatures[val_indexes])
        temperatures[test_indexes] = temperature_scaler(temperatures[test_indexes])

        solubility_means = solubility_vars = None
        if SCALE_TARGETS:
            solubilities[train_indexes], solubility_means, solubility_vars = standard_scale(solubilities[train_indexes])
            target_scaler = partial(standard_scale, means=solubility_means, variances=solubility_vars)
            solubilities[val_indexes] = target_scaler(solubilities[val_indexes])
            solubilities[test_indexes] = target_scaler(solubilities[test_indexes])

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
            batch_size=10_000,
        )

        # initialize the model and train/test
        model = fastpropSolubility(
            **model_kwargs,
            target_means=solubility_means,
            target_vars=solubility_vars,
            solute_means=solute_feature_means,
            solute_vars=solute_feature_vars,
            solvent_means=solvent_feature_means,
            solvent_vars=solvent_feature_vars,
            temperature_means=temperature_means,
            temperature_vars=temperature_vars,
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test(_output_dir, model, train_dataloader, val_dataloader, test_dataloader, 100, 10)
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())
    if remove_output:
        shutil.rmtree(_output_dir)
    return validation_results_df, test_results_df


if __name__ == "__main__":
    train_ensemble(
        remove_output=False,
        num_solute_layers=5,
        solute_hidden_size=1_600,
        num_solvent_layers=5,
        solvent_hidden_size=1_600,
        num_interaction_layers=2,
        interaction_hidden_size=1_800,
        interaction_operation="subtraction",
        activation_fxn="leakyrelu",
        num_features=1613,
        learning_rate=0.0001,
    )