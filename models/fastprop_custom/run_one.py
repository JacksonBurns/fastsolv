import glob
import logging
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import torch
from astartes import train_val_test_split
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import _init_loggers, init_logger
from fastprop.defaults import ALL_2D
from fastprop.metrics import SCORE_LOOKUP, mean_absolute_error_score, root_mean_squared_error_loss, weighted_mean_absolute_percentage_error_score
from fastprop.model import train_and_test
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from sklearn.preprocessing import QuantileTransformer
from torchmetrics.functional.regression import r2_score as tm_r2_score

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)


NUM_REPLICATES = 1
SCALE_TARGETS = True
SHOW_PLOTS = True
SOLUTE_EXTRAPOLATION = True
RANDOM_SEED = 1701  # the final frontier

SOLUTE_COLUMNS: list[str] = ["solute_" + d for d in ALL_2D]
SOLVENT_SOLUMNS: list[str] = ["solvent_" + d for d in ALL_2D]


def impute_missing(data: torch.Tensor, means: torch.Tensor = None):
    return_stats = False
    if means is None:
        return_stats = True
        means = data.nanmean(dim=0)
    # treat missing features' means as zero
    torch.nan_to_num_(means)

    # set missing rows with their column's average value
    data = data.where(~data.isnan(), means)
    if return_stats:
        return data, means
    else:
        return data


def quantile_transform(features: torch.Tensor, prefit: QuantileTransformer = None):
    if prefit:
        return torch.tensor(prefit.transform(features.numpy(force=True)))
    features_np = features.numpy(force=True)
    clf = QuantileTransformer(output_distribution="uniform")
    transformed_features = torch.tensor(clf.fit_transform(features_np))
    return transformed_features, clf


def logS_within_0_7_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(0.7).count_nonzero() / prediction.size(dim=0)


def logS_within_1_0_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(1.0).count_nonzero() / prediction.size(dim=0)


def r2_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    if SHOW_PLOTS and len(prediction) > 128:
        plt.scatter(truth.numpy(force=True), prediction.numpy(force=True), alpha=0.1)
        plt.xlabel("truth")
        plt.ylabel("prediction")
        plt.ylim((-10, 2))
        plt.xlim((-10, 2))
        plt.plot([-10, 2], [-10, 2], color="black", linestyle="-")
        plt.plot([-10, 2], [-9, 3], color="red", linestyle="--", alpha=0.25)
        plt.plot([-10, 2], [-11, 1], color="red", linestyle="--", alpha=0.25)
        plt.show()
    return tm_r2_score(prediction.squeeze(), truth.squeeze())


SCORE_LOOKUP["regression"] = (
    logS_within_0_7_percentage,
    logS_within_1_0_percentage,
    root_mean_squared_error_loss,
    mean_absolute_error_score,
    weighted_mean_absolute_percentage_error_score,
    r2_score,
)


def run_one(data=None, remove_output=False, run_holdout=False, **model_kwargs):
    # setup logging and output directories
    os.makedirs("output", exist_ok=True)
    _init_loggers("output")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(RANDOM_SEED)
    _data_dir = Path("../../data")
    # load the training data
    if data is None:
        df = pd.read_csv(_data_dir / Path("vermeire/prepared_data.csv"), index_col=0)
        smiles_df = df[["solute_smiles", "solvent_smiles"]]
        solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
        temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)
    else:
        solute_features, solvent_features, temperatures, solubilities, smiles_df = data

    if run_holdout:
        # load the holdout data
        df = pd.read_csv(_data_dir / Path("boobier/acetone_solubility_data_features.csv"), index_col=0)
        acetone_solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        acetone_temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        acetone_solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        acetone_solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)
        df = pd.read_csv(_data_dir / Path("boobier/benzene_solubility_data_features.csv"), index_col=0)
        benzene_solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        benzene_temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        benzene_solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        benzene_solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)
        df = pd.read_csv(_data_dir / Path("boobier/ethanol_solubility_data_features.csv"), index_col=0)
        ethanol_solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        ethanol_temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        ethanol_solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        ethanol_solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)

        # load the other holdout data
        df = pd.read_csv(_data_dir / Path("llompart/llompart_features.csv"), index_col=0)
        llompart_solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        llompart_temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        llompart_solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        llompart_solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)

        # load the other other holdout data
        df = pd.read_csv(_data_dir / Path("krasnov/bigsol_features.csv"), index_col=0)
        bigsol_solubilities = torch.tensor(df["logS"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        bigsol_temperatures = torch.tensor(df["temperature"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        bigsol_solute_features = torch.tensor(df[SOLUTE_COLUMNS].to_numpy(), dtype=torch.float32)
        bigsol_solvent_features = torch.tensor(df[SOLVENT_SOLUMNS].to_numpy(), dtype=torch.float32)

        acetone_results, benzene_results, ethanol_results = [], [], []
        llompart_results = []
        bigsol_results = []

    logger.info("Run 'tensorboard --logdir output/tensorboard_logs' to track training progress.")
    random_seed = RANDOM_SEED
    all_test_results, all_validation_results = [], []
    for replicate_number in range(NUM_REPLICATES):
        logger.info(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")

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
        if run_holdout:
            acetone_solute_features = solute_scaler(acetone_solute_features)
            benzene_solute_features = solute_scaler(benzene_solute_features)
            ethanol_solute_features = solute_scaler(ethanol_solute_features)
            llompart_solute_features = solute_scaler(llompart_solute_features)
            bigsol_solute_features = solute_scaler(bigsol_solute_features)

        solvent_features[train_indexes], solvent_feature_means, solvent_feature_vars = standard_scale(solvent_features[train_indexes])
        solvent_scaler = partial(standard_scale, means=solvent_feature_means, variances=solute_feature_vars)
        solvent_features[val_indexes] = solvent_scaler(solvent_features[val_indexes])
        solvent_features[test_indexes] = solvent_scaler(solvent_features[test_indexes])
        if run_holdout:
            acetone_solvent_features = solvent_scaler(acetone_solvent_features)
            benzene_solvent_features = solvent_scaler(benzene_solvent_features)
            ethanol_solvent_features = solvent_scaler(ethanol_solvent_features)
            llompart_solvent_features = solvent_scaler(llompart_solvent_features)
            bigsol_solvent_features = solvent_scaler(bigsol_solvent_features)

        temperatures[train_indexes], temperature_means, temperature_vars = standard_scale(temperatures[train_indexes])
        temperature_scaler = partial(standard_scale, means=temperature_means, variances=temperature_vars)
        temperatures[val_indexes] = temperature_scaler(temperatures[val_indexes])
        temperatures[test_indexes] = temperature_scaler(temperatures[test_indexes])
        if run_holdout:
            acetone_temperatures = temperature_scaler(acetone_temperatures)
            benzene_temperatures = temperature_scaler(benzene_temperatures)
            ethanol_temperatures = temperature_scaler(ethanol_temperatures)
            llompart_temperatures = temperature_scaler(llompart_temperatures)
            bigsol_temperatures = temperature_scaler(bigsol_temperatures)

        solubility_means = solubility_vars = None
        if SCALE_TARGETS:
            solubilities[train_indexes], solubility_means, solubility_vars = standard_scale(solubilities[train_indexes])
            target_scaler = partial(standard_scale, means=solubility_means, variances=solubility_vars)
            solubilities[val_indexes] = target_scaler(solubilities[val_indexes])
            solubilities[test_indexes] = target_scaler(solubilities[test_indexes])
            if run_holdout:
                acetone_solubilities = target_scaler(acetone_solubilities)
                benzene_solubilities = target_scaler(benzene_solubilities)
                ethanol_solubilities = target_scaler(ethanol_solubilities)
                llompart_solubilities = target_scaler(llompart_solubilities)
                bigsol_solubilities = target_scaler(bigsol_solubilities)

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
        if run_holdout:
            acetone_dataloader = fastpropDataLoader(
                SolubilityDataset(
                    acetone_solute_features,
                    acetone_solvent_features,
                    acetone_temperatures,
                    acetone_solubilities,
                ),
                batch_size=10_000,
            )
            benzene_dataloader = fastpropDataLoader(
                SolubilityDataset(
                    benzene_solute_features,
                    benzene_solvent_features,
                    benzene_temperatures,
                    benzene_solubilities,
                ),
                batch_size=10_000,
            )
            ethanol_dataloader = fastpropDataLoader(
                SolubilityDataset(
                    ethanol_solute_features,
                    ethanol_solvent_features,
                    ethanol_temperatures,
                    ethanol_solubilities,
                ),
                batch_size=10_000,
            )
            llompart_dataloader = fastpropDataLoader(
                SolubilityDataset(
                    llompart_solute_features,
                    llompart_solvent_features,
                    llompart_temperatures,
                    llompart_solubilities,
                ),
                batch_size=10_000,
            )
            bigsol_dataloader = fastpropDataLoader(
                SolubilityDataset(
                    bigsol_solute_features,
                    bigsol_solvent_features,
                    bigsol_temperatures,
                    bigsol_solubilities,
                ),
                batch_size=10_000,
            )

        # initialize the model and train/test
        model = fastpropSolubility(
            **model_kwargs,
            target_means=solubility_means,
            target_vars=solubility_vars,
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test("output", model, train_dataloader, val_dataloader, test_dataloader, 100, 10)
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])

        if run_holdout:
            trainer = Trainer(logger=False)
            checkpoints_list = glob.glob(os.path.join("output", "checkpoints", "*.ckpt"))
            latest_file = max(checkpoints_list, key=os.path.getctime)
            model = fastpropSolubility.load_from_checkpoint(latest_file)
            result = trainer.test(model, acetone_dataloader, verbose=False)
            acetone_results.append(result[0])
            result = trainer.test(model, benzene_dataloader, verbose=False)
            benzene_results.append(result[0])
            result = trainer.test(model, ethanol_dataloader, verbose=False)
            ethanol_results.append(result[0])
            result = trainer.test(model, llompart_dataloader, verbose=False)
            llompart_results.append(result[0])
            result = trainer.test(model, bigsol_dataloader, verbose=False)
            bigsol_results.append(result[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())
    if run_holdout:
        holdout_results_df = pd.DataFrame.from_records(acetone_results)
        logger.info("Displaying acetone holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
        holdout_results_df = pd.DataFrame.from_records(benzene_results)
        logger.info("Displaying benzene holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
        holdout_results_df = pd.DataFrame.from_records(ethanol_results)
        logger.info("Displaying ethanol holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
        holdout_results_df = pd.DataFrame.from_records(llompart_results)
        logger.info("Displaying llompart holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
        holdout_results_df = pd.DataFrame.from_records(bigsol_results)
        logger.info("Displaying BigSolDB holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
    if remove_output:
        shutil.rmtree("output")
    return validation_results_df, test_results_df


if __name__ == "__main__":
    run_one(
        run_holdout=True,
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
        learning_rate=0.001,
    )
