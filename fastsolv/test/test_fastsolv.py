import argparse
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from fastsolv import fastsolv
from fastsolv._cli import _fastsolv_predict


class _CSVManager:
    input = "temp_in.csv"
    output = "temp_out.csv"

    def setUp(self):
        with open(self.input, "w") as file:
            file.write("solvent_smiles,solute_smiles,temperature\nCO,C1C=CC=CC1,298\n")

    def tearDown(self):
        Path(self.input).unlink(missing_ok=True)
        Path(self.output).unlink(missing_ok=True)


class TestCLI(_CSVManager, unittest.TestCase):
    def test_output_specified(self):
        with mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(input=self.input, output=self.output, version=None)):
            _fastsolv_predict()


class TestModule(_CSVManager, unittest.TestCase):
    def test_module(self):
        df = pd.read_csv(self.input)
        out = fastsolv(df)
        assert out.any().any()
