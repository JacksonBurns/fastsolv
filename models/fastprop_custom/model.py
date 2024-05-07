"""
model.py - solubility prediction model definition

batches are organized (solute, solvent, temperature)
"""

from typing import Literal

import torch
from fastprop.model import fastprop as _fastprop


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
        interaction_operation: Literal["concatenation", "multiplication", "subtraction"] = "concatenation",
        num_features: int = 1613,
        num_interaction_layers: int = 0,
        learning_rate: float = 0.001,
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
        )
        del self.fnn
        del self.readout

        # solute
        solute_modules = []
        for i in range(num_solute_representation_layers):  # hidden layers
            solute_modules.append(torch.nn.Linear(num_features, num_features))
            solute_modules.append(torch.nn.ReLU())

        # solvent
        solvent_modules = []
        for i in range(num_solvent_representation_layers):  # hidden layers
            solvent_modules.append(torch.nn.Linear(num_features, num_features))
            solvent_modules.append(torch.nn.ReLU())

        # assemble modules (if empty, just passes input through)
        self.solute_representation_module = torch.nn.Sequential(*solute_modules)
        self.solvent_representation_module = torch.nn.Sequential(*solvent_modules)

        # interaction module
        interaction_modules = []
        if interaction_operation == "concatenation":  # size doubles if concatenated
            num_interaction_features = 2 * num_features + 1  # plus temperature
            interaction_modules.append(Concatenation())
        else:
            num_interaction_features = num_features + 1  # plus temperature
            if interaction_operation == "multiplication":
                interaction_modules.append(Multiplication())
            elif interaction_operation == "subtraction":
                interaction_modules.append(Subtraction())
            else:
                raise TypeError(f"Unknown interaction operation '{interaction_operation}'!")
        for i in range(num_interaction_layers):  # hidden layers
            interaction_modules.append(torch.nn.Linear(num_interaction_features, num_interaction_features))
            interaction_modules.append(torch.nn.ReLU())
        self.interaction_module = torch.nn.Sequential(*interaction_modules)

        # readout
        self.readout = torch.nn.Linear(num_interaction_features, 1)
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

    model = fastpropSolubility(2, 1, "multiplication", 1_613, 2, 1e-3)
    print(model)
    print(model(batch))
