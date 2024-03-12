# usage: bash chemprop_train.sh
# trains a comparison model against fastprop

chemprop_train \
--data_path targets.csv \
--smiles_columns solvent_smiles solute_smiles \
--number_of_molecules 2 \
--features_path chemprop_features.csv \
--target_columns logS \
--dataset_type regression \
--save_dir chemprop_highsol_training \
--epochs 50 \
--split_sizes 0.8 0.1 0.1 \
--metric rmse \
--extra_metrics mae r2 \
--batch_size 1024 \
--seed 0
