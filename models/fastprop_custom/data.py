import torch
from torch.utils.data import Dataset as TorchDataset


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
