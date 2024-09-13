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
from astartes import train_test_split
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

from classes import SolubilityDataset, fastpropSolubility

logger = init_logger(__name__)

NUM_REPLICATES = 4
SPLIT_TYPE = "source"  # solute, random
RANDOM_SEED = 3511  # the final frontier
TRAINING_FPATH = Path("krasnov/bigsoldb_fastprop_nonaq.csv")

SOLUTE_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D]
SOLVENT_COLUMNS: list[str] = ["solvent_" + d for d in ALL_2D]


def _f(r):
    if len(r["logS"]) == 1:
        return [np.nan]
    sorted_idxs = np.argsort(r["temperature"])
    unsort_idxs = np.argsort(sorted_idxs)
    # mask out enormous (non-physical) values, negative values, and nan/inf
    grads = [
        i if (np.isfinite(i) and np.abs(i) < 1.0 and i > 0.0) else np.nan
        for i in np.gradient(
            [r["logS"][i] for i in sorted_idxs],
            [r["temperature"][i] for i in sorted_idxs],
        )
    ]
    return [grads[i] for i in unsort_idxs]


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


def train_ensemble(*, data=None, remove_output=False, training_percent=None, **model_kwargs):
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
        metadata_df_og = df[["solute_smiles", "solvent_smiles", "source"]]
        solubilities_og = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
        temperatures_og = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        solute_features_og = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        solvent_features_og = torch.tensor(df[SOLVENT_COLUMNS].to_numpy(), dtype=torch.float32)
    else:
        solute_features_og, solvent_features_og, temperatures_og, solubilities_og, metadata_df_og = data

    random_seed = RANDOM_SEED
    if training_percent is not None:
        logger.warning(f"Down-sampling training data to {training_percent:.2%} size!")
        downsample_df = metadata_df_og.copy()
        downsample_df["original_index"] = np.arange(len(metadata_df_og))
        downsample_df = downsample_df.groupby(["solute_smiles", "solvent_smiles", "source"]).aggregate(list)
        downsample_df = downsample_df.sample(frac=training_percent, replace=False, random_state=random_seed)
        chosen_indexes = downsample_df.explode("original_index")["original_index"].to_numpy().flatten().astype(int)
        logger.warning(f"Actual downsample percentage is {len(chosen_indexes)/len(metadata_df_og):.2%}, count: {len(chosen_indexes)}!")
        metadata_df_og = metadata_df_og.iloc[chosen_indexes]
        metadata_df_og.reset_index(inplace=True, drop=True)
        solubilities_og = solubilities_og[chosen_indexes]
        temperatures_og = temperatures_og[chosen_indexes]
        solute_features_og = solute_features_og[chosen_indexes]
        solvent_features_og = solvent_features_og[chosen_indexes]

    logger.info(f"Run 'tensorboard --logdir {_output_dir}/tensorboard_logs' to track training progress.")
    all_validation_results = []
    for replicate_number in range(NUM_REPLICATES):
        logger.info(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")
        # keep backups so repeat trials don't rescale already scaled data
        metadata_df = metadata_df_og.copy()
        solubilities = solubilities_og.detach().clone()
        temperatures = temperatures_og.detach().clone()
        solute_features = solute_features_og.detach().clone()
        solvent_features = solvent_features_og.detach().clone()

        # split the data s.t. model only sees a subset of the studies used to aggregate the training data
        if SPLIT_TYPE == "source":
            studies_train, studies_val = train_test_split(pd.unique(metadata_df["source"]), random_state=random_seed, train_size=0.90, test_size=0.10)
            train_indexes = metadata_df.index[metadata_df["source"].isin(studies_train)].tolist()
            val_indexes = metadata_df.index[metadata_df["source"].isin(studies_val)].tolist()
        elif SPLIT_TYPE == "solute":
            solutes_train, solutes_val = train_test_split(pd.unique(metadata_df["solute_smiles"]), random_state=random_seed)
            train_indexes = metadata_df.index[metadata_df["solute_smiles"].isin(solutes_train)].tolist()
            val_indexes = metadata_df.index[metadata_df["solute_smiles"].isin(solutes_val)].tolist()
        else:
            train_indexes, val_indexes = train_test_split(np.arange(len(metadata_df)), random_state=random_seed)
        _total = len(metadata_df)
        logger.info(
            f"train: {len(train_indexes)} ({len(train_indexes)/_total:.0%}) validation:" f"{len(val_indexes)} ({len(val_indexes)/_total:.0%})"
        )
        # scaling
        solute_features[train_indexes], solute_feature_means, solute_feature_vars = standard_scale(solute_features[train_indexes])
        solute_scaler = partial(standard_scale, means=solute_feature_means, variances=solute_feature_vars)
        solute_features[val_indexes] = solute_scaler(solute_features[val_indexes])

        solvent_features[train_indexes], solvent_feature_means, solvent_feature_vars = standard_scale(solvent_features[train_indexes])
        solvent_scaler = partial(standard_scale, means=solvent_feature_means, variances=solute_feature_vars)
        solvent_features[val_indexes] = solvent_scaler(solvent_features[val_indexes])

        temperatures[train_indexes], temperature_means, temperature_vars = standard_scale(temperatures[train_indexes])
        temperature_scaler = partial(standard_scale, means=temperature_means, variances=temperature_vars)
        temperatures[val_indexes] = temperature_scaler(temperatures[val_indexes])

        solubilities[train_indexes], solubility_means, solubility_vars = standard_scale(solubilities[train_indexes])
        target_scaler = partial(standard_scale, means=solubility_means, variances=solubility_vars)
        solubilities[val_indexes] = target_scaler(solubilities[val_indexes])

        # calculate the expected gradients post-scaling
        # start by inserting the rescaled data back into the metatdata dataframe _in the right order_
        tgrads = pd.concat(
            (
                metadata_df,
                pd.DataFrame(
                    {
                        "logS": np.ravel(solubilities.numpy()),
                        "temperature": np.ravel(temperatures.numpy()),
                        "source_index": np.arange(len(temperatures)),
                    }
                ),
            ),
            axis=1,
        )
        # group the data by experiment
        tgrads = tgrads.groupby(["source", "solvent_smiles", "solute_smiles"])[["logS", "temperature", "source_index"]].aggregate(list)
        # calculate the gradient at each measurement of logS wrt temperature
        tgrads["logSgradT"] = tgrads.apply(_f, axis=1)
        # get them in the same order as the source data
        tgrads = tgrads.explode(["logSgradT", "source_index"]).sort_values(by="source_index")
        # convert and mask
        tgrads = tgrads["logSgradT"].to_numpy(dtype=np.float32)
        _mask = np.isnan(tgrads)
        logger.warning(f"Masking {np.count_nonzero(_mask)} of {len(_mask)} gradients!")
        logger.info(f"{np.count_nonzero(tgrads > 0)} of {len(tgrads)} were positive!")
        tgrads = torch.tensor(tgrads, dtype=torch.float32).unsqueeze(-1)

        train_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[train_indexes],
                solvent_features[train_indexes],
                temperatures[train_indexes],
                solubilities[train_indexes],
                tgrads[train_indexes],
            ),
            batch_size=256,
            shuffle=True,
            drop_last=bool(int(os.environ.get("ENABLE_REGULARIZATION", 0))),
        )
        val_dataloader = fastpropDataLoader(
            SolubilityDataset(
                solute_features[val_indexes],
                solvent_features[val_indexes],
                temperatures[val_indexes],
                solubilities[val_indexes],
                tgrads[val_indexes],
            ),
            batch_size=4096,
        )
        test_dataloader = fastpropDataLoader(SolubilityDataset([], [], [], [], []))

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
        test_results, validation_results = train_and_test(
            _output_dir,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            100,
            20,
            quiet=remove_output,
            inference_mode=False,
        )
        all_validation_results.append(validation_results[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    if remove_output:
        shutil.rmtree(_output_dir)
    return validation_results_df


# open the output directory, rename the most recent subdir with a new name
def rename_recent_dir(updated_name):
    parent_dir = "output"
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    most_recent_dir = max(subdirs, key=os.path.getmtime)
    new_name = os.path.join(parent_dir, updated_name)
    os.rename(most_recent_dir, new_name)


if __name__ == "__main__":
    # optimized fastprop model
    # run with: DISABLE_CUSTOM_LOSS=1
    # hopt_params = {
    #     "input_activation": "tanh",
    #     "activation_fxn": "relu",
    #     "hidden_size": 3000,
    #     "num_layers": 4,
    #     "solvent_hidden_size": 1000,
    #     "solvent_layers": 2,
    #     "solute_hidden_size": 1200,
    #     "solute_layers": 3,
    # }
    # optimized fastprop_phys model
    # disable custom loss
    # hopt_params = {
    #     "input_activation": "clamp3",
    #     "activation_fxn": "relu",
    #     "hidden_size": 2000,
    #     "num_layers": 2,
    #     "solvent_hidden_size": 2400,
    #     "solvent_layers": 4,
    #     "solute_hidden_size": 400,
    #     "solute_layers": 1,
    # }
    # optimized gradpropphys model
    # enable custom loss
    # hopt_params = {
    #     "input_activation": "clamp3",
    #     "activation_fxn": "relu",
    #     "hidden_size": 2000,
    #     "num_layers": 2,
    #     "solvent_hidden_size": 2400,
    #     "solvent_layers": 4,
    #     "solute_hidden_size": 400,
    #     "solute_layers": 1,
    # }
    # optimized fastprop-sobolev model
    # run with: DISABLE_CUSTOM_LOSS=0
    hopt_params = {
        "input_activation": "clamp3",
        "activation_fxn": "leakyrelu",
        "hidden_size": 3000,
        "num_layers": 2,
    }
    for training_count in (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215):
        training_percent = training_count / 5215
        train_ensemble(
            training_percent=training_percent,
            remove_output=False,
            num_features=1613,
            learning_rate=0.0001,
            **hopt_params,
        )
        rename_recent_dir(f"fastprop_{training_count}")
