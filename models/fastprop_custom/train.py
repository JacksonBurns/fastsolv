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
from fastprop.metrics import (
    SCORE_LOOKUP,
    mean_absolute_error_score,
    root_mean_squared_error_loss,
    weighted_mean_absolute_percentage_error_score,
    r2_score,
)
from fastprop.model import train_and_test
from lightning.pytorch import seed_everything

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)

NUM_REPLICATES = 4
SPLIT_TYPE = "source"  # solute, random
RANDOM_SEED = 1701  # the final frontier
TRAINING_FPATH = Path("krasnov/bigsoldb_downsample.csv")

SOLUTE_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS: list[str] = ["solvent_" + d for d in ALL_2D]


def _f(r):
    if len(r["logS"]) == 1:
        return np.array([0.0])
    return [i if np.isfinite(i) else 0.0 for i in np.gradient(r["logS"], r["temperature"])]


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
    plt.plot([min_val, max_val], [min_val + 1, max_val + 1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val - 1, max_val - 1], color="red", linestyle="--", alpha=0.25)
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


def train_ensemble(data=None, remove_output=False, **model_kwargs):
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
        metadata_df = df[["solute_smiles", "solvent_smiles", "source"]]
        solubilities_og = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
        temperatures_og = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        solute_features_og = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        solvent_features_og = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)
    else:
        solute_features_og, solvent_features_og, temperatures_og, solubilities_og, metadata_df = data

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

        # split the data s.t. model only sees a subset of the studies used to aggregate the training data
        if SPLIT_TYPE == "source":
            studies_train, studies_val, studies_test = train_val_test_split(pd.unique(metadata_df["source"]), random_state=random_seed)
            train_indexes = metadata_df.index[metadata_df["source"].isin(studies_train)].tolist()
            val_indexes = metadata_df.index[metadata_df["source"].isin(studies_val)].tolist()
            test_indexes = metadata_df.index[metadata_df["source"].isin(studies_test)].tolist()
        elif SPLIT_TYPE == "solute":
            solutes_train, solutes_val, solutes_test = train_val_test_split(pd.unique(metadata_df["solute_smiles"]), random_state=random_seed)
            train_indexes = metadata_df.index[metadata_df["solute_smiles"].isin(solutes_train)].tolist()
            val_indexes = metadata_df.index[metadata_df["solute_smiles"].isin(solutes_val)].tolist()
            test_indexes = metadata_df.index[metadata_df["solute_smiles"].isin(solutes_test)].tolist()
        else:
            train_indexes, val_indexes, test_indexes = train_val_test_split(np.arange(len(metadata_df)), random_state=random_seed)
        _total = len(metadata_df)
        logger.info(
            f"train: {len(train_indexes)} ({len(train_indexes)/_total:.0%}) validation:"
            f"{len(val_indexes)} ({len(val_indexes)/_total:.0%}) test: {len(test_indexes)} ({len(test_indexes)/_total:.0%})"
        )
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

        solubilities[train_indexes], solubility_means, solubility_vars = standard_scale(solubilities[train_indexes])
        target_scaler = partial(standard_scale, means=solubility_means, variances=solubility_vars)
        solubilities[val_indexes] = target_scaler(solubilities[val_indexes])
        solubilities[test_indexes] = target_scaler(solubilities[test_indexes])

        # calculate the expected gradients post-scaling
        # start by inserting the rescaled data back into the metatdata dataframe _in the right order_
        tgrads = pd.concat(
            (
                metadata_df,
                pd.DataFrame(
                    {
                        "logS": np.ravel(solubilities.numpy()),
                        "temperature": np.ravel(temperatures.numpy()),
                    }
                ),
            ),
            axis=1,
        )
        # group the data by experiment
        tgrads = tgrads.groupby(["source", "solvent_smiles", "solute_smiles"])[["logS", "temperature"]].aggregate(list)
        # calculate the gradient at each measurement of logS wrt temperature
        tgrads["logSgradT"] = tgrads.apply(_f, axis=1)
        tgrads = tgrads.explode("logSgradT")["logSgradT"].to_numpy(dtype=np.float32)
        logger.info(f"{np.count_nonzero(tgrads > 0)} of {len(tgrads)} were positive!")
        tgrads = torch.tensor(tgrads, dtype=torch.float32)
        temperatures.requires_grad_(True)

        train_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[train_indexes],
                solvent_features[train_indexes],
                temperatures[train_indexes],
                solubilities[train_indexes],
                tgrads[train_indexes],
            ),
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[val_indexes],
                solvent_features[val_indexes],
                temperatures[val_indexes],
                solubilities[val_indexes],
                tgrads[val_indexes],
            ),
        )
        test_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[test_indexes],
                solvent_features[test_indexes],
                temperatures[test_indexes],
                solubilities[test_indexes],
                tgrads[test_indexes],
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
        test_results, validation_results = train_and_test(_output_dir, model, train_dataloader, val_dataloader, test_dataloader, 100, 20)
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
    hopt_params = {
        "input_activation": "clamp3",
        "activation_fxn": "relu",
        "interaction_hidden_size": 2600,
        "num_interaction_layers": 2,
        "interaction_operation": "concatenation",
        "num_solute_layers": 1,
        "solute_hidden_size": 200,
        "num_solvent_layers": 1,
        "solvent_hidden_size": 200,
    }
    train_ensemble(
        remove_output=False,
        num_features=1613,
        learning_rate=0.001,
        **hopt_params,
    )
