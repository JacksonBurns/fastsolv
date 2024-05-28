import torch
from torch.utils.data import Dataset as TorchDataset


class SolubilityDataset(TorchDataset):
    def __init__(
        self,
        solute_features: torch.Tensor,
        solvent_features: torch.Tensor,
        temperature: torch.Tensor,
        is_water: torch.Tensor,
        solubility: torch.Tensor,
    ):
        self.solute_features = solute_features
        self.solvent_features = solvent_features
        self.temperature = temperature
        self.is_water = is_water
        self.solubility = solubility
        self.length = len(solubility)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            (
                self.solute_features[index],
                self.solvent_features[index],
                self.temperature[index],
                self.is_water[index],
            ),
            self.solubility[index],
        )
