import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astartes import train_test_split
from lightning import pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from chemprop import data as chemprop_data_utils
from chemprop import featurizers, nn
from chemprop.models import multi
from chemprop.nn import metrics

NUM_REPLICATES = 4
RANDOM_SEED = 1701  # the final frontier
TRAINING_FPATH = Path("krasnov/bigsoldb_chemprop.csv")


def train_ensemble(*, training_percent=None, **model_kwargs):
    # setup logging and output directories
    _output_dir = Path(f"output/chemprop_{int(datetime.datetime.now(datetime.UTC).timestamp())}")
    os.makedirs(_output_dir, exist_ok=True)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(RANDOM_SEED)
    _data_dir = Path("../../data")

    random_seed = RANDOM_SEED
    all_validation_results = []
    for replicate_number in range(NUM_REPLICATES):
        print(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")
        # load the training data
        df = pd.read_csv(_data_dir / TRAINING_FPATH, index_col=0)

        if training_percent is not None:
            print(f"Down-sampling training data to {training_percent:.2%} size!")
            downsample_df = df.copy()
            downsample_df["original_index"] = np.arange(len(df))
            downsample_df = downsample_df.groupby(["solute_smiles", "solvent_smiles", "source"]).aggregate(list)
            downsample_df = downsample_df.sample(frac=training_percent, replace=False, random_state=random_seed)
            chosen_indexes = downsample_df.explode("original_index")["original_index"].to_numpy().flatten().astype(int)
            print(f"Actual downsample percentage is {len(chosen_indexes)/len(df):.2%}, count: {len(chosen_indexes)}!")
            df = df.iloc[chosen_indexes]
            df.reset_index(inplace=True, drop=True)

        all_data = [
            [
                chemprop_data_utils.MoleculeDatapoint.from_smi(smi, [log_s], x_d=np.array([temperature]))
                for smi, log_s, temperature in zip(df["solute_smiles"], df["logS"], df["temperature"])
            ],
            list(map(chemprop_data_utils.MoleculeDatapoint.from_smi, df["solvent_smiles"])),
        ]

        # split the data s.t. model only sees a subset of the studies used to aggregate the training data
        studies_train, studies_val = train_test_split(pd.unique(df["source"]), random_state=random_seed, train_size=0.90, test_size=0.10)
        train_indexes = df.index[df["source"].isin(studies_train)].tolist()
        val_indexes = df.index[df["source"].isin(studies_val)].tolist()

        _total = len(df)
        print(f"train: {len(train_indexes)} ({len(train_indexes)/_total:.0%}) validation:" f"{len(val_indexes)} ({len(val_indexes)/_total:.0%})")

        train_data, val_data, _ = chemprop_data_utils.split_data_by_indices(all_data, train_indices=train_indexes, val_indices=val_indexes)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_datasets = [chemprop_data_utils.MoleculeDataset(train_data[i], featurizer) for i in range(len(all_data))]
        val_datasets = [chemprop_data_utils.MoleculeDataset(val_data[i], featurizer) for i in range(len(all_data))]
        train_mcdset = chemprop_data_utils.MulticomponentDataset(train_datasets)
        train_mcdset.cache = True
        scaler = train_mcdset.normalize_targets()
        extra_datapoint_descriptors_scaler = train_mcdset.normalize_inputs("X_d")
        val_mcdset = chemprop_data_utils.MulticomponentDataset(val_datasets)
        val_mcdset.cache = True
        val_mcdset.normalize_targets(scaler)
        val_mcdset.normalize_inputs("X_d", extra_datapoint_descriptors_scaler)

        train_loader = chemprop_data_utils.build_dataloader(train_mcdset)
        val_loader = chemprop_data_utils.build_dataloader(val_mcdset, shuffle=False)

        # build Chemprop
        mcmp = nn.MulticomponentMessagePassing(
            blocks=[nn.BondMessagePassing() for _ in range(len(all_data))],
            n_components=len(all_data),
        )
        agg = nn.NormAggregation()
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(
            input_dim=mcmp.output_dim + 1,  # temperature
            hidden_dim=1_400,
            n_layers=3,
            output_transform=output_transform,
        )
        X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_datapoint_descriptors_scaler[0])
        metric_list = [metrics.MSEMetric(), metrics.RMSEMetric(), metrics.MAEMetric(), metrics.R2Metric()]
        mcmpnn = multi.MulticomponentMPNN(
            mcmp,
            agg,
            ffn,
            batch_norm=False,
            metrics=metric_list,
            X_d_transform=X_d_transform,
        )
        print(mcmpnn)
        try:
            repetition_number = len(os.listdir(os.path.join(_output_dir, "tensorboard_logs"))) + 1
        except FileNotFoundError:
            repetition_number = 1
        tensorboard_logger = TensorBoardLogger(
            _output_dir,
            name="tensorboard_logs",
            version=f"repetition_{repetition_number}",
            default_hp_metric=False,
        )
        callbacks = [
            EarlyStopping(
                monitor="val/mse",
                mode="min",
                verbose=False,
                patience=10,
            ),
            ModelCheckpoint(
                monitor="val/mse",
                dirpath=os.path.join(_output_dir, "checkpoints"),
                filename=f"repetition-{repetition_number}" + "-{epoch:02d}",
                save_top_k=1,
                mode="min",
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=40,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
        )

        trainer.fit(mcmpnn, train_loader, val_loader)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(f"Reloading best model from checkpoint file: {ckpt_path}")
        mcmpnn = mcmpnn.__class__.load_from_checkpoint(ckpt_path)
        val_results = trainer.validate(mcmpnn, val_loader)
        all_validation_results.append(val_results[0])
        random_seed += 1
        # ensure that the model is re-instantiated
        del mcmpnn

    validation_results_df = pd.DataFrame.from_records(all_validation_results)
    print("Displaying validation results:\n", validation_results_df.describe().transpose().to_string())
    return validation_results_df


# open the output directory, rename the most recent subdir with a new name
def rename_recent_dir(updated_name):
    parent_dir = "output"
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    most_recent_dir = max(subdirs, key=os.path.getmtime)
    new_name = os.path.join(parent_dir, updated_name)
    os.rename(most_recent_dir, new_name)


if __name__ == "__main__":
    for training_count in (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215):
        training_percent = training_count / 5215
        train_ensemble(training_percent=training_percent)
        rename_recent_dir(f"chemprop_{training_count}")
