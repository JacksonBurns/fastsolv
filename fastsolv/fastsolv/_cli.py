import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset

from fastprop.data import clean_dataset, fastpropDataLoader
from fastprop.defaults import DESCRIPTOR_SET_LOOKUP, init_logger
from fastprop.descriptors import get_descriptors
from fastprop.io import load_saved_descriptors
from fastprop.model import fastprop
import argparse
import datetime
import sys
from importlib.metadata import version
from time import perf_counter
from rdkit import Chem
import warnings

import yaml

from fastprop.defaults import ALL_2D
from pathlib import Path

from fastprop import DEFAULT_TRAINING_CONFIG
from fastprop.defaults import init_logger


from ._module import fastsolv


def _fastsolv_predict(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="fastsolv command line interface - try 'fastsolv --help'")
        parser.add_argument("-v", "--version", help="print version and exit", action="store_true")
        parser.add_argument("input", nargs=1, help="CSV file containing 'solvent_smiles', 'solute_smiles', and 'temperature' in Kelvin.")
        parser.add_argument("-o", "--output", help="Name of output file", default="fastsolv_output.csv")
    
    args = parser.parse_args()
    if args.version:
        print(version("fastsolv"))
        exit(0)

    df = pd.read_csv(args.input)
    out = fastsolv(df)
    out.to_csv(args.output)
