"""
model.py - solubility prediction model definition

batches are organized (solute, solvent, temperature)
"""

from typing import Literal

import torch
from fastprop.data import inverse_standard_scale, standard_scale
from fastprop.model import fastprop as _fastprop

ENABLE_SNN = False
ENABLE_DROPOUT = False
ENABLE_BATCHNORM = True


class Concatenation(torch.nn.Module):
    def forward(self, batch):
        return torch.cat(batch, dim=1)


class Multiplication(torch.nn.Module):
    def forward(self, batch):
        return torch.cat((batch[0] * batch[1], batch[2]), dim=1)


class Subtraction(torch.nn.Module):
    def forward(self, batch):
        return torch.cat((batch[0] - batch[1], batch[2]), dim=1)


# just had a lecture about transformers, this is kinda like something from that, i guess (not really)
class PairwiseMax(torch.nn.Module):
    def forward(self, batch):
        pairswise_intxns = torch.matmul(batch[0].unsqueeze(2), batch[1].unsqueeze(1))
        maxes, _ = pairswise_intxns.max(dim=2)
        return torch.cat((maxes, batch[2]), dim=1)


class ReLUn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n = torch.nn.Parameter(torch.randn(()))

    def forward(self, batch: torch.Tensor):
        return torch.nn.functional.relu(batch).minimum(self.n)


class WideTanh(torch.nn.Module):
    def __init__(self, width: float = 3.14) -> None:
        super().__init__()
        self.width = width

    def forward(self, batch: torch.Tensor):
        return torch.nn.functional.tanh(batch / self.width)


# inputs are normally distributed (by our scaling) so clamping at n
# is like applying a n-sigma cutoff, i.e. anything else is an outlier
class ClampN(torch.nn.Module):
    def __init__(self, n: float) -> None:
        super().__init__()
        self.n = n

    def forward(self, batch: torch.Tensor):
        return torch.clamp(batch, min=-self.n, max=self.n)

    def extra_repr(self) -> str:
        return f"n={self.n}"


def _build_mlp(input_size, hidden_size, act_fun, num_layers):
    modules = []
    for i in range(num_layers):
        modules.append(torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        if (num_layers == 1) or (i < num_layers - 1):  # no activation after last layer, unless perceptron
            if ENABLE_SNN:
                modules.append(torch.nn.SELU())
                if ENABLE_DROPOUT:
                    modules.append(torch.nn.AlphaDropout())
            else:
                if act_fun == "sigmoid":
                    modules.append(torch.nn.Sigmoid())
                elif act_fun == "tanh":
                    modules.append(torch.nn.Tanh())
                elif act_fun == "relu":
                    modules.append(torch.nn.ReLU())
                elif act_fun == "relu6":
                    modules.append(torch.nn.ReLU6())
                elif act_fun == "leakyrelu":
                    modules.append(torch.nn.LeakyReLU())
                elif act_fun == "relun":
                    modules.append(ReLUn())
                else:
                    raise TypeError(f"What is {act_fun}?")
                if ENABLE_DROPOUT:
                    modules.append(torch.nn.Dropout())
                if ENABLE_BATCHNORM:
                    modules.append(torch.nn.BatchNorm1d(hidden_size))
    return modules


class fastpropSolubility(_fastprop):
    def __init__(
        self,
        num_solute_layers: int = 0,
        solute_hidden_size: int = 1_000,
        num_solvent_layers: int = 0,
        solvent_hidden_size: int = 1_000,
        num_interaction_layers: int = 0,
        interaction_hidden_size: int = 1_000,
        interaction_operation: Literal["concatenation", "multiplication", "subtraction", "pairwisemax"] = "concatenation",
        activation_fxn: Literal["relu", "relu6", "sigmoid", "leakyrelu", "relun", "tanh"] = "relu6",
        input_activation: Literal["sigmoid", "tanh", "clamp3"] = None,
        num_features: int = 1613,
        learning_rate: float = 0.0001,
        target_means: torch.Tensor = None,
        target_vars: torch.Tensor = None,
        solute_means: torch.Tensor = None,
        solute_vars: torch.Tensor = None,
        solvent_means: torch.Tensor = None,
        solvent_vars: torch.Tensor = None,
        temperature_means: torch.Tensor = None,
        temperature_vars: torch.Tensor = None,
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

        # for later predicting
        self.register_buffer("solute_means", solute_means)
        self.register_buffer("solute_vars", solute_vars)
        self.register_buffer("solvent_means", solvent_means)
        self.register_buffer("solvent_vars", solvent_vars)
        self.register_buffer("temperature_means", temperature_means)
        self.register_buffer("temperature_vars", temperature_vars)

        # solute - temperature is concatenated to the input features
        solute_modules = _build_mlp(num_features + 1, solute_hidden_size, activation_fxn, num_solute_layers)
        solute_hidden_size = num_features + 1 if num_solute_layers == 0 else solute_hidden_size

        # solvent - temperature is concatenated to the input features
        solvent_modules = _build_mlp(num_features + 1, solvent_hidden_size, activation_fxn, num_solvent_layers)
        solvent_hidden_size = num_features + 1 if num_solvent_layers == 0 else solvent_hidden_size

        # optionally bound input
        if input_activation == "clamp3":
            solute_modules.insert(0, ClampN(n=3.0))
            solvent_modules.insert(0, ClampN(n=3.0))
        elif input_activation == "sigmoid":
            solute_modules.insert(0, torch.nn.Sigmoid())
            solvent_modules.insert(0, torch.nn.Sigmoid())
        elif input_activation == "tanh":
            solute_modules.insert(0, torch.nn.Tanh())
            solvent_modules.insert(0, torch.nn.Tanh())

        # assemble modules (if empty, just passes input through)
        self.solute_representation_module = torch.nn.Sequential(*solute_modules)
        self.solvent_representation_module = torch.nn.Sequential(*solvent_modules)

        # interaction module
        interaction_modules = []
        if interaction_operation == "concatenation":  # size increases if concatenated
            num_interaction_features = solvent_hidden_size + solute_hidden_size + 1  # plus temperature
            interaction_modules.append(Concatenation())
        elif interaction_operation == "pairwisemax":
            num_interaction_features = solute_hidden_size + 1
            interaction_modules.append(PairwiseMax())
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
        interaction_modules += _build_mlp(num_interaction_features, interaction_hidden_size + 1, activation_fxn, num_interaction_layers)
        self.interaction_module = torch.nn.Sequential(*interaction_modules)

        # readout
        self.readout = torch.nn.Linear(num_interaction_features if num_interaction_layers == 0 else interaction_hidden_size + 1, 1)
        self.save_hyperparameters()

    def forward(self, batch):
        solute_features, solvent_features, temperature = batch
        solute_representation = self.solute_representation_module(torch.cat((solute_features, temperature), dim=1))
        solvent_representation = self.solvent_representation_module(torch.cat((solvent_features, temperature), dim=1))
        output = self.interaction_module((solute_representation, solvent_representation, temperature))
        y_hat = self.readout(output)
        return y_hat

    def predict_step(self, batch):
        err_msg = ""
        for stat_obj, stat_name in zip(
            (
                self.solute_means,
                self.solute_vars,
                self.solvent_means,
                self.solvent_vars,
                self.temperature_means,
                self.temperature_vars,
                self.target_means,
                self.target_vars,
            ),
            (
                "solute_means",
                "solute_vars",
                "solvent_means",
                "solvent_vars",
                "temperature_means",
                "temperature_vars",
                "target_means",
                "target_vars",
            ),
        ):
            if stat_obj is None:
                err_msg.append(f"{stat_name} is None!\n")
        if err_msg:
            raise RuntimeError("Missing scaler statistics!\n" + err_msg)

        solute_features, solvent_features, temperature = batch[0]  # batch 1 is solubility
        solute_features = standard_scale(solute_features, self.solute_means, self.solute_vars)
        solvent_features = standard_scale(solvent_features, self.solvent_means, self.solvent_vars)
        temperature = standard_scale(temperature, self.temperature_means, self.temperature_vars)
        with torch.inference_mode():
            logits = self.forward((solute_features, solvent_features, temperature))
        return inverse_standard_scale(logits, self.target_means, self.target_vars)


if __name__ == "__main__":
    # test batch of 4
    solute = torch.rand((4, 100))
    solvent = torch.rand((4, 100))
    temperature = torch.rand((4, 1))
    batch = (solute, solvent, temperature)

    model = fastpropSolubility(
        num_solute_layers=3,
        solute_hidden_size=1_400,
        num_solvent_layers=1,
        solvent_hidden_size=200,
        num_interaction_layers=2,
        interaction_hidden_size=2_400,
        interaction_operation="pairwisemax",
        activation_fxn="relu",
        input_activation="clamp3",
        num_features=100,
        learning_rate=0.01,
        target_means=None,
        target_vars=None,
    )
    print(model)
    print(model(batch))
