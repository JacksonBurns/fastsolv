"""
model.py - solubility prediction model definition

batches are organized (solute, solvent, temperature)
"""

import os
from types import SimpleNamespace
from typing import Literal

import torch
from fastprop.data import inverse_standard_scale, standard_scale
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


class Addition(torch.nn.Module):
    def forward(self, batch):
        return torch.cat((batch[0] + batch[1], batch[2]), dim=1)


# just had a lecture about transformers, this is kinda like something from that, i guess (not really)
class PairwiseMax(torch.nn.Module):
    def forward(self, batch):
        pairswise_intxns = torch.matmul(batch[0].unsqueeze(2), batch[1].unsqueeze(1))
        maxes, _ = pairswise_intxns.max(dim=2)
        return torch.cat((maxes, batch[2]), dim=1)


class IndependentVariableAppendingSequential(torch.nn.Sequential):
    def forward(self, input):
        embedding, independent_variable = input
        for module in self:
            if isinstance(module, torch.nn.Linear):
                embedding = torch.cat((embedding, independent_variable), dim=1)
            embedding = module(embedding)
        return embedding


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
        modules.append(torch.nn.Linear(1 + (input_size if i == 0 else hidden_size), hidden_size))
        if (num_layers == 1) or (i < num_layers - 1):  # no activation after last layer, unless perceptron
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
            else:
                raise TypeError(f"What is {act_fun}?")
            if int(os.environ.get("ENABLE_REGULARIZATION", 0)):
                modules.append(torch.nn.BatchNorm1d(hidden_size))
                modules.append(torch.nn.Dropout())
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
        interaction_operation: Literal[
            "concatenation",
            "multiplication",
            "subtraction",
            "addition",
        ] = "concatenation",
        activation_fxn: Literal["relu", "leakyrelu"] = "relu",
        input_activation: Literal["sigmoid", "tanh", "clamp3"] = "sigmoid",
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
        # spoof the readout size
        self.readout = SimpleNamespace(out_features=1)

        # for later predicting
        self.register_buffer("solute_means", solute_means)
        self.register_buffer("solute_vars", solute_vars)
        self.register_buffer("solvent_means", solvent_means)
        self.register_buffer("solvent_vars", solvent_vars)
        self.register_buffer("temperature_means", temperature_means)
        self.register_buffer("temperature_vars", temperature_vars)

        # solute - temperature is concatenated to the input features
        solute_modules = _build_mlp(num_features, solute_hidden_size, activation_fxn, num_solute_layers)
        solute_hidden_size = solute_hidden_size if num_solute_layers > 0 else num_features

        # solvent - temperature is concatenated to the input features
        solvent_modules = _build_mlp(num_features, solvent_hidden_size, activation_fxn, num_solvent_layers)
        solvent_hidden_size = solvent_hidden_size if num_solvent_layers > 0 else num_features

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
        self.solute_representation_module = IndependentVariableAppendingSequential(*solute_modules)
        self.solvent_representation_module = IndependentVariableAppendingSequential(*solvent_modules)

        # interaction module
        interaction_modules = []
        if interaction_operation == "concatenation":  # size increases if concatenated
            num_interaction_features = solvent_hidden_size + solute_hidden_size
            interaction_modules.append(Concatenation())
        else:
            if solute_hidden_size != solvent_hidden_size:
                raise TypeError(
                    f"Invalid choice of interaction ({interaction_operation}) for mis-matched solute/solvent"
                    f" embedding sizes {solute_hidden_size}/{solvent_hidden_size}."
                )
            num_interaction_features = solvent_hidden_size
            if interaction_operation == "multiplication":
                interaction_modules.append(Multiplication())
            elif interaction_operation == "subtraction":
                interaction_modules.append(Subtraction())
            elif interaction_operation == "addition":
                interaction_modules.append(Addition())
            else:
                raise TypeError(f"Unknown interaction operation '{interaction_operation}'!")
        self.interaction_module = torch.nn.Sequential(*interaction_modules)

        # solution
        solution_modules = _build_mlp(num_interaction_features, interaction_hidden_size, activation_fxn, num_interaction_layers)
        # readout
        solution_modules.append(torch.nn.Linear(1 + (num_interaction_features if num_interaction_layers == 0 else interaction_hidden_size), 1))
        self.solution_module = IndependentVariableAppendingSequential(*solution_modules)

        self.save_hyperparameters()

    def forward(self, batch):
        solute_features, solvent_features, temperature = batch
        solute_representation = self.solute_representation_module((solute_features, temperature))
        solvent_representation = self.solvent_representation_module((solvent_features, temperature))
        solution_representation = self.interaction_module((solute_representation, solvent_representation))
        y_hat = self.solution_module((solution_representation, temperature))
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

    @torch.enable_grad()
    def _custom_loss(self, batch: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], name: str):
        (_solute, _solvent, temperature), y, y_grad = batch
        temperature.requires_grad_()
        y_hat: torch.Tensor = self.forward((_solute, _solvent, temperature))
        y_loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        (y_grad_hat,) = torch.autograd.grad(
            y_hat,
            temperature,
            grad_outputs=torch.ones_like(y_hat),
            retain_graph=True,
        )
        _scale_factor = 100.0
        y_grad_loss = _scale_factor * (y_grad_hat - y_grad).pow(2).nanmean()  # MSE ignoring nan
        loss = y_loss + y_grad_loss
        self.log(f"{name}_{self.training_metric}_scaled_loss", loss)
        self.log(f"{name}_logS_scaled_loss", y_loss)
        self.log(f"{name}_dlogSdT_scaled_loss", y_grad_loss)
        return loss, y_hat

    def _plain_loss(self, batch: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], name: str):
        (_solute, _solvent, temperature), y, y_grad = batch
        y_hat: torch.Tensor = self.forward((_solute, _solvent, temperature))
        loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        self.log(f"{name}_{self.training_metric}_scaled_loss", loss)
        return loss, y_hat

    def _loss(self, batch: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], name: str):
        if int(os.environ.get("DISABLE_CUSTOM_LOSS", 0)):
            return self._plain_loss(batch, name)
        else:
            return self._custom_loss(batch, name)

    def training_step(self, batch, batch_idx):
        return self._loss(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._loss(batch, "validation")
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._loss(batch, "test")
        self._human_loss(y_hat, batch, "test")
        return loss


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
        activation_fxn="relu6",
        num_features=100,
        learning_rate=0.01,
        target_means=None,
        target_vars=None,
    )
    print(model)
    print(model(batch))
