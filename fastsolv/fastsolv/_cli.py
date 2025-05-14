import argparse
import sys
from importlib.metadata import version

import pandas as pd

from ._module import fastsolv


def _fastsolv_predict(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="fastsolv solid solubility predictor CLI.",
            epilog="Example: fastsolv input.csv -o predictions.csv",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "-v", "--version",
            action="store_true",
            help="Print the version of fastsolv and exit."
        )
        parser.add_argument(
            "input",
            nargs='?',
            default=None,
            help="Path to the input CSV file. Required unless --version is used. "
                 "The CSV must contain 'solvent_smiles', 'solute_smiles', and 'temperature' (in Kelvin) columns."
        )
        parser.add_argument(
            "-o", "--output",
            default="fastsolv_output.csv",
            help="Path to save the output CSV file."
        )

    args = parser.parse_args()

    if args.version:
        print(version("fastsolv"))
        sys.exit(0)

    # If not asking for version, input file is mandatory
    if args.input is None:
        parser.error("the following arguments are required: input (path to CSV file)")

    required_columns = ['solvent_smiles', 'solute_smiles', 'temperature']

    try:
        df = pd.read_csv(args.input)

        if df.empty:
            print(f"Error: The input file '{args.input}' is empty.")
            sys.exit(1)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: The input file '{args.input}' is missing the following required columns: {', '.join(missing_columns)}.")
            sys.exit(1)

        print(f"Processing '{args.input}'...")
        out_df = fastsolv(df)
        out_df.to_csv(args.output, index=False)
        print(f"Predictions saved to '{args.output}'.")

    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file '{args.input}'. Please ensure it is a valid CSV.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
