import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astartes import train_test_split
from chemprop import data as chemprop_data_utils
from chemprop import featurizers, nn
from chemprop.models import multi
from chemprop.nn import metrics
from lightning import pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler

NUM_REPLICATES = 4
RANDOM_SEED = 1701  # the final frontier
TRAINING_FPATH = Path("krasnov/bigsoldb_chemprop_nonaq.csv")


class CustomMSEMetric(metrics.MSEMetric):
    def forward(self, preds, targets, mask, weights, lt_mask, gt_mask):
        return torch.nn.functional.mse_loss(preds, targets[:, 0, None], reduction="mean")


class SobolevMulticomponentMPNN(multi.MulticomponentMPNN):
    def training_step(self, batch, batch_idx):
        return self._sobolev_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._sobolev_loss(batch, "val")

    @torch.enable_grad()
    def _sobolev_loss(self, batch, name):
        bmg, V_d, X_d, targets, *_ = batch
        # track grad for temperature
        X_d.requires_grad_()
        Z = self.fingerprint(bmg, V_d, X_d)
        y_hat = self.predictor.train_step(Z)
        y_loss = torch.nn.functional.mse_loss(y_hat, targets[:, 0, None], reduction="mean")
        (y_grad_hat,) = torch.autograd.grad(
            y_hat,
            X_d,
            grad_outputs=torch.ones_like(y_hat),
            retain_graph=True,
        )
        _scale_factor = 1.0
        y_grad_loss = _scale_factor * (y_grad_hat - targets[:, 1]).pow(2).nanmean()  # MSE ignoring nan
        loss = y_loss + y_grad_loss
        self.log(f"{name}/sobolev_loss", loss, batch_size=len(batch[0]))
        self.log(f"{name}/logs_loss", y_loss, batch_size=len(batch[0]))
        self.log(f"{name}/grad_loss", y_grad_loss, batch_size=len(batch[0]))
        self.log(f"{name}_loss", loss, prog_bar=True, batch_size=len(batch[0]))
        return loss


def _f(r):
    if len(r["scaled_logS"]) == 1:
        return [np.nan]
    sorted_idxs = np.argsort(r["scaled_temperature"])
    unsort_idxs = np.argsort(sorted_idxs)
    # mask out enormous (non-physical) values, negative values, and nan/inf
    grads = [
        i if (np.isfinite(i) and np.abs(i) < 1.0 and i > 0.0) else np.nan
        for i in np.gradient(
            [r["scaled_logS"][i] for i in sorted_idxs],
            [r["scaled_temperature"][i] for i in sorted_idxs],
        )
    ]
    return [grads[i] for i in unsort_idxs]


def train_ensemble(*, training_percent=None, **model_kwargs):
    # setup logging and output directories
    _output_dir = Path(f"output/chemprop_{int(datetime.datetime.now(datetime.UTC).timestamp())}")
    os.makedirs(_output_dir, exist_ok=True)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    seed_everything(RANDOM_SEED)
    _data_dir = Path("../../data")

    random_seed = RANDOM_SEED
    # load the training data
    src_df = pd.read_csv(_data_dir / TRAINING_FPATH, index_col=0)
    if training_percent is not None:
        print(f"Down-sampling training data to {training_percent:.2%} size!")
        downsample_df = src_df.copy()
        downsample_df["original_index"] = np.arange(len(src_df))
        downsample_df = downsample_df.groupby(["solute_smiles", "solvent_smiles", "source"]).aggregate(list)
        downsample_df = downsample_df.sample(frac=training_percent, replace=False, random_state=random_seed)
        chosen_indexes = downsample_df.explode("original_index")["original_index"].to_numpy().flatten().astype(int)
        print(f"Actual downsample percentage is {len(chosen_indexes)/len(src_df):.2%}, count: {len(chosen_indexes)}!")
        src_df = src_df.iloc[chosen_indexes]
        src_df.reset_index(inplace=True, drop=True)

    all_validation_results = []
    for replicate_number in range(NUM_REPLICATES):
        print(f"Training model {replicate_number+1} of {NUM_REPLICATES} ({random_seed=})")
        df = src_df.copy()
        # split the data s.t. model only sees a subset of the studies used to aggregate the training data
        studies_train, studies_val = train_test_split(pd.unique(df["source"]), random_state=random_seed, train_size=0.90, test_size=0.10)
        train_indexes = df.index[df["source"].isin(studies_train)].tolist()
        val_indexes = df.index[df["source"].isin(studies_val)].tolist()

        # manual re-scaling
        target_scaler = StandardScaler().fit(df[["logS"]].iloc[train_indexes])
        scaled_logs = target_scaler.transform(df[["logS"]]).ravel()
        temperature_scaler = StandardScaler().fit(df[["temperature"]].iloc[train_indexes])
        scaled_temperature = temperature_scaler.transform(df[["temperature"]]).ravel()

        # calculate known temperature gradients
        tgrads = pd.concat(
            (
                df,
                pd.DataFrame(
                    {
                        "source_index": np.arange(len(df["temperature"])),
                        "scaled_temperature": scaled_temperature,
                        "scaled_logS": scaled_logs,
                    }
                ),
            ),
            axis=1,
        )
        # group the data by experiment
        tgrads = tgrads.groupby(["source", "solvent_smiles", "solute_smiles"])[["scaled_logS", "scaled_temperature", "source_index"]].aggregate(list)
        # calculate the gradient at each measurement of logS wrt temperature
        tgrads["logSgradT"] = tgrads.apply(_f, axis=1)
        # get them in the same order as the source data
        tgrads = tgrads.explode(["logSgradT", "source_index"]).sort_values(by="source_index")
        # convert and mask
        tgrads = tgrads["logSgradT"].to_numpy(dtype=np.float32)
        _mask = np.isnan(tgrads)
        print(f"Masking {np.count_nonzero(_mask)} of {len(_mask)} gradients!")
        print(f"{np.count_nonzero(tgrads > 0)} of {len(tgrads)} were positive!")
        all_data = [
            [
                chemprop_data_utils.MoleculeDatapoint.from_smi(smi, [log_s, log_s_grad_T], x_d=np.array([temperature]))
                for smi, log_s, log_s_grad_T, temperature in zip(df["solute_smiles"], scaled_logs, tgrads, df["temperature"])
            ],
            list(map(chemprop_data_utils.MoleculeDatapoint.from_smi, df["solvent_smiles"])),
        ]

        _total = len(df)
        print(f"train: {len(train_indexes)} ({len(train_indexes)/_total:.0%}) validation:" f"{len(val_indexes)} ({len(val_indexes)/_total:.0%})")

        train_data, val_data, _ = chemprop_data_utils.split_data_by_indices(all_data, train_indices=train_indexes, val_indices=val_indexes)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_datasets = [chemprop_data_utils.MoleculeDataset(train_data[i], featurizer) for i in range(len(all_data))]
        val_datasets = [chemprop_data_utils.MoleculeDataset(val_data[i], featurizer) for i in range(len(all_data))]
        train_mcdset = chemprop_data_utils.MulticomponentDataset(train_datasets)
        train_mcdset.normalize_inputs("X_d", [temperature_scaler, None])
        train_mcdset.cache = True
        val_mcdset = chemprop_data_utils.MulticomponentDataset(val_datasets)
        # chemprop docs say to do this, but it actually gets scaled during training
        # val_mcdset.normalize_inputs("X_d", [temperature_scaler, None])
        val_mcdset.cache = True

        train_loader = chemprop_data_utils.build_dataloader(train_mcdset, batch_size=256, num_workers=1, persistent_workers=True)
        val_loader = chemprop_data_utils.build_dataloader(val_mcdset, batch_size=1_024, shuffle=False, num_workers=1, persistent_workers=True)

        # build Chemprop
        mcmp = nn.MulticomponentMessagePassing(
            blocks=[nn.BondMessagePassing(depth=3, d_h=800) for _ in range(len(all_data))],
            n_components=len(all_data),
        )
        agg = nn.MeanAggregation()
        output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
        ffn = nn.RegressionFFN(
            input_dim=mcmp.output_dim + 1,  # temperature
            hidden_dim=800,
            n_layers=2,
            criterion=CustomMSEMetric(),
            output_transform=output_transform,
        )
        X_d_transform = nn.ScaleTransform.from_standard_scaler(temperature_scaler)
        metric_list = [CustomMSEMetric()]
        mcmpnn = SobolevMulticomponentMPNN(
            mcmp,
            agg,
            ffn,
            batch_norm=True,
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
                monitor="val_loss",
                mode="min",
                verbose=False,
                patience=15,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=os.path.join(_output_dir, "checkpoints"),
                filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.02f}",
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
            # to enable sobolev loss during validation
            inference_mode=False,
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
    train_ensemble(training_percent=1.0)
    # for training_count in (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215):
    #     training_percent = training_count / 5215
    #     train_ensemble(training_percent=training_percent)
    #     rename_recent_dir(f"chemprop_{training_count}")
