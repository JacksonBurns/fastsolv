import logging
import os

import pandas as pd
import torch
from astartes import train_val_test_split
from fastprop.data import fastpropDataLoader, standard_scale
from fastprop.defaults import _init_loggers, init_logger
from fastprop.model import train_and_test
from lightning.pytorch import seed_everything

from data import SolubilityDataset
from model import fastpropSolubility

logger = init_logger(__name__)


def main():
    # setup logging and output directories
    os.makedirs("output_optimized", exist_ok=True)
    _init_loggers("output_optimized")
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
    random_seed = 42
    all_test_results, all_validation_results = [], []
    for replicate_number in range(3):
        logger.info(f"Training model {replicate_number+1} of 3 ({random_seed=})")

        # split the data s.t. some solutes are not seen during training
        solutes_train, solutes_val, solutes_test = train_val_test_split(pd.unique(solute_df["solute_smiles"]), random_state=random_seed)
        train_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_train)].tolist()
        val_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_val)].tolist()
        test_indexes = solute_df.index[solute_df["solute_smiles"].isin(solutes_test)].tolist()
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
            0,
            0,
            "concatenation",
            1_613,
            2,
            0.001,
            3000,
            solubility_means,
            solubility_vars,
        )
        logger.info("Model architecture:\n{%s}", str(model))
        test_results, validation_results = train_and_test(
            "output_optimized",
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
