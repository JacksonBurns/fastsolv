#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


# In[2]:


df = pl.read_csv(Path("../data/krasnov/bigsoldb_downsample.csv"), columns=["solute_smiles", "solvent_smiles", "logS", "temperature", "source"]).to_pandas()
df = df.groupby(["source", "solvent_smiles", "solute_smiles"])[["logS", "temperature"]].aggregate(list)
df


# In[3]:


def _f(r):
    try:
        grads = np.gradient(r["logS"], r["temperature"])
        mean_grad = np.mean(grads)
        if any(grads > mean_grad*3):
            return np.nan
        return mean_grad
    except:
        return np.nan


# In[4]:


df["mean_grad"] = df.apply(_f, axis=1)
df


# In[5]:


df = df[np.isfinite(df["mean_grad"])]


# In[12]:


count = 0
seen = set()
# for (i, row_i) in df.itertuples(index=True):
for idx_i, (row_i, logS_i, temperature_i, mean_grad_i) in enumerate(df.itertuples(index=True)):
    for idx_j, (row_j, logS_j, temperature_j, mean_grad_j) in enumerate(df.itertuples(index=True)):
        if row_i == row_j:
            continue
        if row_i[1] != row_j[1]:  # skip different solvent
            continue
        if row_i[2] == row_j[2]:  # skip same solute
            continue
        seen.add((idx_i, idx_j))
        if (idx_j, idx_i) in seen:
            continue
        if len(logS_i) < 10 or len(logS_j) < 10:
            continue
        # mistmatched gradients
        if np.abs(mean_grad_j - mean_grad_i) < 0.04:
            continue
        if np.abs(logS_i[0] - logS_j[1]) < 1:
            continue
        count += 1
        print(row_i, row_j)
        fig, axes_dict = plt.subplot_mosaic(
            """
        AABC
        AADE
        """,
            figsize=(14, 7),
        )
        axes_dict['A'].scatter(x=temperature_i, y=logS_i, color='b')
        axes_dict['A'].scatter(x=temperature_j, y=logS_j, color='r')
        axes_dict["A"].set_xlabel("Temperature (K)")
        axes_dict["A"].set_ylabel("logS")
        axes_dict["B"].imshow(Draw.MolToImage(Chem.MolFromSmiles(row_i[1]), size=(400, 400)))
        axes_dict["B"].set_title("Solvent", color='b')
        axes_dict["B"].set_axis_off()
        axes_dict["C"].imshow(Draw.MolToImage(Chem.MolFromSmiles(row_i[2]), size=(400, 400)))
        axes_dict["C"].set_title("Solute", color='b')
        axes_dict["C"].set_axis_off()
        axes_dict["D"].imshow(Draw.MolToImage(Chem.MolFromSmiles(row_j[1]), size=(400, 400)))
        axes_dict["D"].set_title("Solvent", color='r')
        axes_dict["D"].set_axis_off()
        axes_dict["E"].imshow(Draw.MolToImage(Chem.MolFromSmiles(row_j[2]), size=(400, 400)))
        axes_dict["E"].set_title("Solute", color='r')
        axes_dict["E"].set_axis_off()
        plt.show()
    #     break
    # else:
    #     continue
    # break
count

