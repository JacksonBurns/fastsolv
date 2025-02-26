# paper

This directory contains the code for model development, training, and analysis.
Additional code for running the Vermeire et. al model is also provided.
This code is primarily intended to aid in reproducing the results of the original study, but it may also be used to train models on new targets or extend the model architecture.

See `sobolev_demo.ipynb` for a worked example of how Sobolev training works on a single-molecule property.

## Installation
Installation should take less than ten minutes.
We suggest using `conda` to manage dependencies, though other solutions such as `venv` or `uv` may also work.
This code requires the following dependencies, all of which are readily installed via `pip` or `conda` with python version 3.9 or newer (3.11 suggested):
```
numpy
pandas
polars
rdkit
scikit-learn
scipy
torch
pytorch-lightning
astartes
fastprop
chemprop>=2,<2.1
```

For hyperparameter optimization specifically, `ray` and `optuna` are also required.

For data munging the Krasnov dataset, the `thermo` package is also required.

Finally for visualizations and analysis, `matplotlib`, `seaborn`, `matplotlib_venn`, `sklearn`, and `scipy` are required.

A graphics processing unit (GPU) is optional for training, but speeds it up significantly and is highly recommended.
Scripts for training `fastsolv` on remote computing resources are included as a demonstration but are not functional for other systems than our own.

We have specifically developed this code using the below versions on ubuntu 22, though all modern versions of the listed dependencies and all operating systems should work (additional dependencies as required by our direct dependencies are also listed):
```
Package            Version 
------------------ --------
asttokens          2.4.1
black              23.12.1
bokeh              3.4.2
certifi            2024.7.4
charset-normalizer 3.3.2
click              8.1.7
comm               0.2.2
contourpy          1.2.1
coverage           7.5.4
coveralls          4.0.1
cycler             0.12.1
dataclasses        0.6
debugpy            1.8.2
decorator          5.1.1
docopt             0.6.2
exceptiongroup     1.2.0
executing          2.0.1
idna               3.7
importlib_metadata 8.0.0
iniconfig          2.0.0
ipykernel          6.29.5
ipython            8.18.1
isort              5.13.2
jedi               0.19.1
Jinja2             3.1.4
joblib             1.4.2
jupyter_client     8.6.2
jupyter_core       5.7.2
kiwisolver         1.4.5
llvmlite           0.43.0
MarkupSafe         2.1.5
matplotlib         3.3.2
matplotlib-inline  0.1.7
matplotlib-venn    0.11.10
mordred            1.2.0
mordredcommunity   2.0.6
mypy-extensions    1.0.0
nest_asyncio       1.6.0
networkx           2.8.8
numba              0.60.0
numpy              1.26.4
packaging          24.1
pandas             2.2.2
parso              0.8.4
pathspec           0.12.1
pexpect            4.9.0
pickleshare        0.7.5
pillow             10.4.0
pip                24.0
platformdirs       4.2.2
pluggy             1.5.0
prompt_toolkit     3.0.47
psutil             6.0.0
ptyprocess         0.7.0
pure-eval          0.2.2
Pygments           2.18.0
pynndescent        0.5.13
pyparsing          3.1.2
pytest             8.2.2
pytest-cov         5.0.0
python-dateutil    2.9.0
pytz               2024.1
PyYAML             6.0.1
pyzmq              26.0.3
rdkit              2024.3.3
requests           2.32.3
scikit-learn       0.24.2
scipy              1.13.1
seaborn            0.11.1
setuptools         70.1.1
shapely            2.0.4
six                1.16.0
stack-data         0.6.2
threadpoolctl      3.5.0
tomli              2.0.1
tornado            6.4.1
tqdm               4.66.4
traitlets          5.14.3
typing_extensions  4.12.2
tzdata             2024.1
umap-learn         0.5.6
urllib3            2.2.2
wcwidth            0.2.13
wheel              0.43.0
wordcloud          1.9.3
xyzservices        2024.6.0
zipp               3.19.2
```

Finally, `solprop` was installed according to these instructions, taken from [this GitHub issue](https://github.com/fhvermei/SolProp_ML/issues/6#issuecomment-1317395817):
```bash
conda create -n solprop -c fhvermei python=3.7 solprop_ml
conda install -n solprop -c rmg descriptastorus
pip install polars  # needed only for our use-case
```

## Usage

For an example of training a single-molecule model with Sobolev training, see `sobolev_demo.ipynb`.

To reproduce the results of the original study, see the below instructions.
The data must be downloaded and munged first using the scripts in `data.py` before running or analyzing either our models or the Vermeire et. al model.
Upon running, scripts will generate the corresponding numerical results (such as RMSE) and/or plots/figures.

To adapt the code in this repository for your own target properties, follow the below workflow but substitute in your data.
For properties defined on pairs of molecules the code will work directly (just keep track of which names are expected for the various inputs) and for single molecule properties see `sobolev_demo.ipynb`.

### Data

The data files are hosted at the external links given at the top of `boobier.py`, `krasnov.py`, and `vermeire.py`.
The data can be prepared for training by first running `python vermeire.py`, then `python krasnov.py`, then `python boobier.py` (this allows us to drop the overlap between these sets).

### SolProp

Install the Vermeire et. al model according to the above instructions and then run `python solprop_test.py`.

### Models

The `models` directory contains the model definitions for the `fastprop`- and chemprop-based `fastsolv` models, as well as a mean-guessing baseline which was not included in the paper.
Each can be executed by running `python train.py` and then `python test.py`.
For the former, the random splitting seed is specified at the top of the file and the very bottom of the file contains the call to `train_ensemble` that actually initiates model training (the generation of the aleatoric error plot data is shown in comments).
For the latter, the filepath to the trained model must be edited at the bottom of the file.
Individual model training takes a few minutes with a GPU-accelerated laptop computer.

### Analysis
The directory contains jupyter notebook which can be opened in Visual Studio Code (or any other similar software) and executed.
They reproduce the figures shown in the paper. 
