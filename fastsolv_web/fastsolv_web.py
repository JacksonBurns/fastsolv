import fcntl
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
from fastsolv import fastsolv

app = Flask(__name__)

logfile = Path('fastsolv_web.log')

if logfile.exists():
    print(f"Output logfile ({logfile}) from previous run found - exiting to avoid overwriting. Please move existing file.")
    exit(1)

with open(logfile, "w") as file:
    file.write("unix_timestamp,solute_smiles,solvent_smiles,temperature,logS_pred,logS_pred_stdev\n")
    
@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = None
    if request.method == 'POST':
        # Retrieve the text box entries and convert them into a 10x3 Pandas DataFrame
        data = []
        for i in range(10):
            row = []
            for j in range(3):
                field_name = f'field_{i}_{j}'
                row.append(request.form.get(field_name))
            data.append(row)
        
        df = pd.DataFrame(data, columns=['solute_smiles', 'solvent_smiles', 'temperature'])
        df.replace("", np.nan, inplace=True)
        df.dropna(inplace=True)
        df['temperature'] = df['temperature'].astype(float)
        result_df = fastsolv(df)
        result_html = result_df.to_html(classes='dataframe table table-striped', index=True)
        
        # dump results to a log file for later analysis
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

    return render_template('index.html', result=result_html)
