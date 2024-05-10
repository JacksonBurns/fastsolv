"""
model.py - solubility prediction model definition

batches are organized (solute, solvent, temperature)
"""

from typing import Literal

import torch
from fastprop.model import fastprop as _fastprop

ENABLE_SNN = False
ENABLE_DROPOUT = False


class Concatenation(torch.nn.Module):
    def forward(self, batch):
        return torch.cat(batch, dim=1)


class Multiplication(torch.nn.Module):
    def forward(self, batch):
        return torch.cat((batch[0] * batch[1], batch[2]), dim=1)


class Subtraction(torch.nn.Module):
    def forward(self, batch):
        return torch.cat((batch[0] - batch[1], batch[2]), dim=1)


class fastpropSolubility(_fastprop):
    def __init__(
        self,
        num_solute_representation_layers: int = 0,
        num_solvent_representation_layers: int = 0,
        branch_hidden_size: int = 1000,
        num_interaction_layers: int = 0,
        interaction_hidden_size: int = 1600,
        interaction_operation: Literal["concatenation", "multiplication", "subtraction"] = "concatenation",
        num_features: int = 1613,
        learning_rate: float = 0.001,
        target_means: torch.Tensor = None,
        target_vars: torch.Tensor = None,
    ):
        super().__init__(
            input_size=num_features,
            hidden_size=1,  # we will overwrite this
            fnn_layers=0,  # and this
            readout_size=1,  # and this
            num_tasks=1,  # actually equal to len(descriptors), but we don't want to see performance on each
            learning_rate=learning_rate,
            problem_type="regression",
            target_names=[],
            target_means=target_means,
            target_vars=target_vars,
        )
        del self.fnn
        del self.readout

        # solute
        solute_modules = []
        for i in range(num_solute_representation_layers):  # hidden layers
            solute_modules.append(torch.nn.Linear(num_features if i == 0 else branch_hidden_size, branch_hidden_size))
            if ENABLE_SNN:
                solute_modules.append(torch.nn.SELU())
                if ENABLE_DROPOUT:
                    solute_modules.append(torch.nn.AlphaDropout())
            else:
                solute_modules.append(torch.nn.ReLU6())
                if ENABLE_DROPOUT:
                    solute_modules.append(torch.nn.Dropout())
        solute_hidden_size = num_features if num_solute_representation_layers == 0 else branch_hidden_size

        # solvent
        solvent_modules = []
        for i in range(num_solvent_representation_layers):  # hidden layers
            solvent_modules.append(torch.nn.Linear(num_features if i == 0 else branch_hidden_size, branch_hidden_size))
            if ENABLE_SNN:
                solvent_modules.append(torch.nn.SELU())
                if ENABLE_DROPOUT:
                    solvent_modules.append(torch.nn.AlphaDropout())
            else:
                solvent_modules.append(torch.nn.ReLU6())
                if ENABLE_DROPOUT:
                    solvent_modules.append(torch.nn.Dropout())
        solvent_hidden_size = num_features if num_solvent_representation_layers == 0 else branch_hidden_size

        # assemble modules (if empty, just passes input through)
        self.solute_representation_module = torch.nn.Sequential(*solute_modules)
        self.solvent_representation_module = torch.nn.Sequential(*solvent_modules)

        # interaction module
        interaction_modules = []
        if interaction_operation == "concatenation":  # size increases if concatenated
            num_interaction_features = solvent_hidden_size + solute_hidden_size + 1  # plus temperature
            interaction_modules.append(Concatenation())
        else:
            if solute_hidden_size != solvent_hidden_size:
                raise TypeError(
                    f"Invalid choice of interaction ({interaction_operation}) for mis-matched solute/solvent"
                    f" embedding sizes {solute_hidden_size}/{solvent_hidden_size}."
                )
            num_interaction_features = solvent_hidden_size + 1  # plus temperature
            if interaction_operation == "multiplication":
                interaction_modules.append(Multiplication())
            elif interaction_operation == "subtraction":
                interaction_modules.append(Subtraction())
            else:
                raise TypeError(f"Unknown interaction operation '{interaction_operation}'!")
        for i in range(num_interaction_layers):  # hidden layers
            interaction_modules.append(torch.nn.Linear(num_interaction_features if i == 0 else interaction_hidden_size + 1, interaction_hidden_size + 1))
            if ENABLE_SNN:
                interaction_modules.append(torch.nn.SELU())
                if ENABLE_DROPOUT:
                    interaction_modules.append(torch.nn.AlphaDropout())
            else:
                interaction_modules.append(torch.nn.ReLU6())
                if ENABLE_DROPOUT:
                    interaction_modules.append(torch.nn.Dropout())
        self.interaction_module = torch.nn.Sequential(*interaction_modules)

        # readout
        self.readout = torch.nn.Linear(num_interaction_features if num_interaction_layers == 0 else interaction_hidden_size + 1, 1)
        self.save_hyperparameters()

    def forward(self, batch):
        solute_features, solvent_features, temperature = batch
        solute_representation = self.solute_representation_module(solute_features)
        solvent_representation = self.solvent_representation_module(solvent_features)
        output = self.interaction_module((solute_representation, solvent_representation, temperature))
        y_hat = self.readout(output)
        return y_hat


if __name__ == "__main__":
    # test batch of 4
    solute = torch.rand((4, 1613))
    solvent = torch.rand((4, 1613))
    temperature = torch.rand((4, 1))
    batch = (solute, solvent, temperature)

    model = fastpropSolubility(2, 1, 1000, 2, 1600, "multiplication", 1_613, 1e-3)
    print(model)
    print(model(batch))
