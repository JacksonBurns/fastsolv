import glob
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from astartes import train_test_split
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import train_and_test
from fastprop.metrics import SCORE_LOOKUP
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)


NUM_REPLICATES = 8


def logS_within_0_7_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(0.7).count_nonzero() / prediction.size(dim=0)


def logS_within_1_0_percentage(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return (truth - prediction).abs().less_equal(1.0).count_nonzero() / prediction.size(dim=0)


SCORE_LOOKUP["regression"] = tuple(list(SCORE_LOOKUP["regression"]) + [logS_within_0_7_percentage, logS_within_1_0_percentage])


def main():
    # setup logging and output directories
    os.makedirs("output_optimized", exist_ok=True)
    _init_loggers("output_optimized")
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(42)

    # load the training data
    df = pd.read_csv("prepared_data.csv", index_col=0)
    solute_df = df[["solute_smiles"]]
    solubilities = torch.tensor(df.iloc[:, 2].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # keep everything 2D
    temperatures = torch.tensor(df.iloc[:, 3].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    solute_features = torch.tensor(df.iloc[:, 4 : (4 + 1_613)].to_numpy(), dtype=torch.float32)
    solvent_features = torch.tensor(df.iloc[:, (4 + 1_613) :].to_numpy(), dtype=torch.float32)

    # load the holdout data
    df = pd.read_csv(Path("holdout_set/acetone_solubility_data_features.csv"), index_col=0)
    acetone_solubilities = torch.tensor(df.iloc[:, 3].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    acetone_temperatures = torch.tensor(df.iloc[:, 2].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    acetone_solute_features = torch.tensor(df.iloc[:, 4 : (4 + 1_613)].to_numpy(), dtype=torch.float32)
    acetone_solvent_features = torch.tensor(df.iloc[:, (4 + 1_613) :].to_numpy(), dtype=torch.float32)
    df = pd.read_csv(Path("holdout_set/benzene_solubility_data_features.csv"), index_col=0)
    benzene_solubilities = torch.tensor(df.iloc[:, 3].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    benzene_temperatures = torch.tensor(df.iloc[:, 2].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    benzene_solute_features = torch.tensor(df.iloc[:, 4 : (4 + 1_613)].to_numpy(), dtype=torch.float32)
    benzene_solvent_features = torch.tensor(df.iloc[:, (4 + 1_613) :].to_numpy(), dtype=torch.float32)
    df = pd.read_csv(Path("holdout_set/ethanol_solubility_data_features.csv"), index_col=0)
    ethanol_solubilities = torch.tensor(df.iloc[:, 3].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    ethanol_temperatures = torch.tensor(df.iloc[:, 2].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    ethanol_solute_features = torch.tensor(df.iloc[:, 4 : (4 + 1_613)].to_numpy(), dtype=torch.float32)
    ethanol_solvent_features = torch.tensor(df.iloc[:, (4 + 1_613) :].to_numpy(), dtype=torch.float32)

    logger.info("Run 'tensorboard --logdir output/tensorboard_logs' to track training progress.")
    random_seed = 42
    solutes_hopt, solutes_test = train_test_split(pd.unique(solute_df["solute_smiles"]), random_state=random_seed)
    test_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_test)].tolist()
    hopt_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_hopt)].tolist()
    all_test_results, all_validation_results = [], []
    acetone_results, benzene_results, ethanol_results = [], [], []
    for replicate_number in range(NUM_REPLICATES):
        logger.info(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")

        # split the data s.t. some solutes are not seen during training
        solutes_train, solutes_val = train_test_split(pd.unique(solute_df["solute_smiles"][hopt_indexes]), random_state=random_seed)
        train_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_train)].tolist()
        val_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_val)].tolist()
        
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

        acetone_solvent_features = standard_scale(acetone_solvent_features, solvent_feature_means, solvent_feature_vars)
        acetone_solute_features = standard_scale(acetone_solute_features, solute_feature_vars, solute_feature_vars)
        acetone_solubilities = standard_scale(acetone_solubilities, solubility_means, solubility_vars)
        acetone_temperatures = standard_scale(acetone_temperatures, temperature_means, temperature_vars)

        benzene_solvent_features = standard_scale(benzene_solvent_features, solvent_feature_means, solvent_feature_vars)
        benzene_solute_features = standard_scale(benzene_solute_features, solute_feature_vars, solute_feature_vars)
        benzene_solubilities = standard_scale(benzene_solubilities, solubility_means, solubility_vars)
        benzene_temperatures = standard_scale(benzene_temperatures, temperature_means, temperature_vars)

        ethanol_solvent_features = standard_scale(ethanol_solvent_features, solvent_feature_means, solvent_feature_vars)
        ethanol_solute_features = standard_scale(ethanol_solute_features, solute_feature_vars, solute_feature_vars)
        ethanol_solubilities = standard_scale(ethanol_solubilities, solubility_means, solubility_vars)
        ethanol_temperatures = standard_scale(ethanol_temperatures, temperature_means, temperature_vars)

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
        acetone_dataloader = fastpropDataLoader(
            SolubilityDataset(
                acetone_solute_features,
                acetone_solvent_features,
                acetone_temperatures,
                acetone_solubilities,
            )
        )
        benzene_dataloader = fastpropDataLoader(
            SolubilityDataset(
                benzene_solute_features,
                benzene_solvent_features,
                benzene_temperatures,
                benzene_solubilities,
            )
        )
        ethanol_dataloader = fastpropDataLoader(
            SolubilityDataset(
                ethanol_solute_features,
                ethanol_solvent_features,
                ethanol_temperatures,
                ethanol_solubilities,
            )
        )

        # initialize the model and train/test
        model = fastpropSolubility(
            num_solute_representation_layers=2,
            num_solvent_representation_layers=1,
            branch_hidden_size=2_000,
            interaction_operation="multiplication",
            target_means=solubility_means,
            target_vars=solubility_vars,
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test("output_optimized", model, train_dataloader, val_dataloader, test_dataloader, 50, 10)
        all_test_results.append(test_results[0])
        all_validation_results.append(validation_results[0])

        trainer = Trainer(logger=False)
        checkpoints_list = glob.glob(os.path.join("output_optimized", "checkpoints", "*.ckpt"))
        latest_file = max(checkpoints_list, key=os.path.getctime)
        model = fastpropSolubility.load_from_checkpoint(latest_file)
        result = trainer.test(model, acetone_dataloader, verbose=False)
        acetone_results.append(result[0])
        result = trainer.test(model, benzene_dataloader, verbose=False)
        benzene_results.append(result[0])
        result = trainer.test(model, ethanol_dataloader, verbose=False)
        ethanol_results.append(result[0])

        random_seed += 1
        # ensure that the model is re-instantiated
        del model

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    logger.info("Displaying validation results:\n%s", validation_results_df.describe().transpose().to_string())
    test_results_df = pd.DataFrame.from_records(all_test_results)
    logger.info("Displaying testing results:\n%s", test_results_df.describe().transpose().to_string())

    holdout_results_df = pd.DataFrame.from_records(acetone_results)
    logger.info("Displaying acetone holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
    holdout_results_df = pd.DataFrame.from_records(benzene_results)
    logger.info("Displaying benzene holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())
    holdout_results_df = pd.DataFrame.from_records(ethanol_results)
    logger.info("Displaying ethanol holdout set results:\n%s", holdout_results_df.describe().transpose().to_string())


if __name__ == "__main__":
    main()
