import os

from train import train_ensemble

training_counts = (20, 50, 100, 200, 500, 1000, 2000, 3500, 5215)

fastprop_config = {
    "input_activation": "sigmoid",
    "activation_fxn": "leakyrelu",
    "interaction_hidden_size": 800,
    "num_interaction_layers": 4,
    "interaction_operation": "concatenation",
    "num_solute_layers": 0,
    "num_solvent_layers": 0,
    "solute_hidden_size": 0,
    "solvent_hidden_size": 0,
}
fastprop_sobolev_config = {
    "input_activation": "clamp3",
    "activation_fxn": "leakyrelu",
    "interaction_hidden_size": 1000,
    "num_interaction_layers": 4,
    "interaction_operation": "concatenation",
    "num_solute_layers": 0,
    "num_solvent_layers": 0,
    "solute_hidden_size": 0,
    "solvent_hidden_size": 0,
}


def rename_recent_dir(updated_name):
    parent_dir = 'output'
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    most_recent_dir = max(subdirs, key=os.path.getmtime)
    new_name = os.path.join(parent_dir, updated_name)
    os.rename(most_recent_dir, new_name)


for training_count in training_counts:
    training_percent = training_count/5215
    os.environ['DISABLE_CUSTOM_LOSS'] = "1"
    train_ensemble(
        remove_output=False,
        training_percent=training_percent,
        num_features=1613,
        learning_rate=0.001,
        **fastprop_config,
    )
    rename_recent_dir(f'fastprop_{training_count}')

    os.environ['DISABLE_CUSTOM_LOSS'] = "0"
    train_ensemble(
        remove_output=False,
        training_percent=training_percent,
        num_features=1613,
        learning_rate=0.001,
        **fastprop_sobolev_config,
    )
    rename_recent_dir(f'fastprop_sobolev_{training_count}')
