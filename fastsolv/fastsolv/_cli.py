import argparse
from importlib.metadata import version

import pandas as pd

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
