from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import torch
from fastsolv import fastsolv

app = Flask(__name__)

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

    return render_template('index.html', result=result_html)
