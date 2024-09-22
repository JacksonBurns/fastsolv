"""
classes.py - solubility prediction model definition and associated classes

batches are organized (solute, solvent, temperature)
"""

import os
from types import SimpleNamespace
from typing import Literal

import torch
from torch.utils.data import Dataset as TorchDataset
from fastprop.data import inverse_standard_scale, standard_scale
from fastprop.model import fastprop as _fastprop


class SolubilityDataset(TorchDataset):
    def __init__(
        self,
        solute_features: torch.Tensor,
        solvent_features: torch.Tensor,
        temperature: torch.Tensor,
        solubility: torch.Tensor,
        solubility_gradient: torch.Tensor,
    ):
        self.solute_features = solute_features
        self.solvent_features = solvent_features
        self.temperature = temperature
        self.solubility = solubility
        self.solubility_gradient = solubility_gradient
        self.length = len(solubility)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            (
                self.solute_features[index],
                self.solvent_features[index],
                self.temperature[index],
            ),
            self.solubility[index],
            self.solubility_gradient[index],
        )


class Concatenation(torch.nn.Module):
    def forward(self, batch):
        return torch.cat(batch, dim=1)


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
        num_layers: 2,
        hidden_size: 1_800,
        activation_fxn: Literal["relu", "leakyrelu"] = "relu",
        input_activation: Literal["sigmoid", "clamp3"] = "sigmoid",
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

        fnn_modules = [Concatenation(), ClampN(n=3.0) if input_activation == "clamp3" else torch.nn.Sigmoid()]
        _input_size = num_features * 2 + 1
        fnn_modules += _build_mlp(_input_size, hidden_size=hidden_size, act_fun=activation_fxn, num_layers=num_layers)
        fnn_modules.append(torch.nn.Linear(hidden_size if num_layers else _input_size, 1))
        self.fnn = torch.nn.Sequential(*fnn_modules)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.fnn(batch)

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
        _scale_factor = 10.0
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


class GradPropPhys(fastpropSolubility):
    def __init__(
        self,
        solvent_layers: 2,
        solvent_hidden_size: 1_800,
        solute_layers: 2,
        solute_hidden_size: 1_800,
        num_layers: 2,
        hidden_size: 1_800,
        activation_fxn: Literal["relu", "leakyrelu"] = "relu",
        input_activation: Literal["sigmoid", "clamp3"] = "sigmoid",
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
            num_layers=2,
            hidden_size=hidden_size,
            activation_fxn=activation_fxn,
            input_activation=input_activation,
            num_features=num_features,
            learning_rate=learning_rate,
            target_means=target_means,
            target_vars=target_vars,
            solute_means=solute_means,
            solute_vars=solute_vars,
            solvent_means=solvent_means,
            solvent_vars=solvent_vars,
            temperature_means=temperature_means,
            temperature_vars=temperature_vars,
        )

        fnn_modules = [Concatenation(), ClampN(n=3.0) if input_activation == "clamp3" else torch.nn.Sigmoid()]
        _input_size = num_features + 1
        fnn_modules += _build_mlp(_input_size, hidden_size=solute_hidden_size, act_fun=activation_fxn, num_layers=solute_layers)
        _solute_out_size = solute_hidden_size if solute_layers else _input_size
        self.solute_fnn = torch.nn.Sequential(*fnn_modules)

        fnn_modules = [Concatenation(), ClampN(n=3.0) if input_activation == "clamp3" else torch.nn.Sigmoid()]
        _input_size = num_features + 1
        fnn_modules += _build_mlp(_input_size, hidden_size=solvent_hidden_size, act_fun=activation_fxn, num_layers=solvent_layers)
        _solvent_out_size = solvent_hidden_size if solvent_layers else _input_size
        self.solvent_fnn = torch.nn.Sequential(*fnn_modules)

        fnn_modules = [Concatenation(), ClampN(n=3.0) if input_activation == "clamp3" else torch.nn.Sigmoid()]
        _input_size = _solute_out_size + _solvent_out_size + 1
        fnn_modules += _build_mlp(_input_size, hidden_size=hidden_size, act_fun=activation_fxn, num_layers=num_layers)
        fnn_modules.append(torch.nn.Linear(hidden_size if num_layers else _input_size, 1))
        self.fnn = torch.nn.Sequential(*fnn_modules)

        self.save_hyperparameters()

    def forward(self, batch):
        (
            solute_features,
            solvent_features,
            temperature,
        ) = batch
        solute_representation = self.solute_fnn((solute_features, temperature))
        solvent_representation = self.solvent_fnn((solvent_features, temperature))
        return self.fnn((solute_representation, solvent_representation, temperature))


if __name__ == "__main__":
    # test batch of 4
    solute = torch.rand((4, 100))
    solvent = torch.rand((4, 100))
    temperature = torch.rand((4, 1))
    batch = (solute, solvent, temperature)

    model = fastpropSolubility()
    print(model)
    print(model(batch))
