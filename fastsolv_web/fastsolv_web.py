import fcntl
import time
from datetime import datetime
from pathlib import Path
import json
import io
import csv

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from fastsolv import fastsolv

app = Flask(__name__)

# Note: The original logfile handling is preserved.
logfile = Path('fastsolv_web.log')

# Check and create the log file if it does not exist
if not logfile.exists():
    with open(logfile, "w") as file:
        file.write("unix_timestamp,solute_smiles,solvent_smiles,temperature,logS_pred,logS_pred_stdev\n")

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Retrieve the SMILES and temperature settings from the form
            solutes = request.form.get('solute_smiles').splitlines()
            solvents = request.form.get('solvent_smiles').splitlines()

            data = []
            temperatures = []

            # Generate the temperature points based on user input
            min_temp = float(request.form.get('min_temp'))
            max_temp = float(request.form.get('max_temp'))
            num_points = int(request.form.get('num_points'))
            
            # Generate linearly spaced temperature points
            temperatures = np.linspace(min_temp, max_temp, num_points)
            
            # Create a comprehensive list of all combinations
            for solute in solutes:
                for solvent in solvents:
                    for temp in temperatures:
                        if solute and solvent:
                            data.append([solute.strip(), solvent.strip(), temp])
            
            # Create a DataFrame for the fastsolv model
            df = pd.DataFrame(data, columns=['solute_smiles', 'solvent_smiles', 'temperature'])

            limit = 5000
            skip_logs = False
            if (n_rows := len(df)) > limit:
                result_df = pd.DataFrame(
                    data={
                        "FAILED": [
                            f"Entered data has {n_rows} calculations to perform, more than the limit of {limit}",
                            "Please remove some solutes, solvents, or temperature points"
                            ]})
                skip_logs = True
            else:
                # Run the prediction
                result_df = fastsolv(df)
            
            # Convert DataFrame to a CSV string (for the text output)
            csv_output = io.StringIO()
            result_df.to_csv(csv_output, index=not skip_logs)
            csv_string = csv_output.getvalue()
            
            # Package the results
            results = {
                'csv': csv_string,
            }

            if not(skip_logs):
                # dump results to a log file for later analysis (original logic)
                start_time = datetime.now().timestamp()
                log_str = "\n".join([
                    str(start_time) + "," + ",".join([str(i) for i in row_index] + row.astype(str).values.tolist())
                    for row_index, row in result_df.iterrows()]
                ) + "\n"
                with open(logfile, 'a') as file:
                    while (datetime.now().timestamp() - start_time) < 2.0:
                        try:
                            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking exclusive lock
                            try:
                                file.write(log_str)
                            finally:
                                fcntl.flock(file, fcntl.LOCK_UN)  # Unlock
                            break  # Exit the loop if write is successful
                        except BlockingIOError:
                            time.sleep(0.1)  # Wait before trying again
                    else:
                        print(f"Unable to aquire lock on {logfile}, dumping log contents:", log_str)

        except Exception as e:
            # In case of an error, we can still render the template with a message.
            print(f"An error occurred: {e}")
            results = None

    return render_template('index.html', results=results)
