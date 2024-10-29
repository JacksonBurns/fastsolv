from pathlib import Path

import numpy as np
import pandas as pd
from astartes import train_test_split

from sklearn.metrics import mean_squared_error

NUM_REPLICATES = 4
TRAINING_FPATH = Path("krasnov/bigsoldb_chemprop_nonaq.csv")


def baseline_ensemble(*, training_percent=None, random_seed=None):
    _data_dir = Path("../../data")

    # load the training data
    src_df = pd.read_csv(_data_dir / TRAINING_FPATH, index_col=0)
    if training_percent is not None:
        downsample_df = src_df.copy()
        downsample_df["original_index"] = np.arange(len(src_df))
        downsample_df = downsample_df.groupby(["solute_smiles", "solvent_smiles", "source"]).aggregate(list)
        downsample_df = downsample_df.sample(frac=training_percent, replace=False, random_state=random_seed)
        chosen_indexes = downsample_df.explode("original_index")["original_index"].to_numpy().flatten().astype(int)
        src_df = src_df.iloc[chosen_indexes]
        src_df.reset_index(inplace=True, drop=True)

    model_mean = 0.0
    for replicate_number in range(NUM_REPLICATES):
        df = src_df.copy()
        studies_train, studies_val = train_test_split(pd.unique(df["source"]), random_state=random_seed, train_size=0.90, test_size=0.10)
        train_indexes = df.index[df["source"].isin(studies_train)].tolist()
        training_mean = df["logS"].iloc[train_indexes].mean()
        model_mean += training_mean
        random_seed += 1
    model_mean /= NUM_REPLICATES

    rmses = []
    for holdout_fpath in (
        Path("boobier/leeds_acetone_chemprop.csv"),
        Path("boobier/leeds_benzene_chemprop.csv"),
        Path("boobier/leeds_ethanol_chemprop.csv"),
        Path("vermeire/solprop_chemprop_nonaq.csv"),
    ):
        # load the holdout data
        df = pd.read_csv(Path("../../data") / holdout_fpath, index_col=0)
        mse = mean_squared_error(np.tile(model_mean, len(df["logS"])), df["logS"])
        rmse = np.sqrt(mse)
        rmses.append(rmse)
    return rmses


if __name__ == "__main__":
    for random_seed in (1337, 1701, 3511):
        baseline_leeds_results = []
        baseline_solprop_results = []
        for training_count in (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215):
            training_percent = training_count / 5215
            leeds_acetone, leeds_benzene, leeds_ethanol, solprop = baseline_ensemble(training_percent=training_percent, random_seed=random_seed)
            baseline_leeds_results.append([leeds_acetone, leeds_benzene, leeds_ethanol])
            baseline_solprop_results.append(solprop)
        print(f"{baseline_leeds_results=}")
        print(f"{baseline_solprop_results=}")
