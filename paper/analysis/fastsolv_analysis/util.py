import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from rdkit import Chem

def inchify(smi):
    return Chem.MolToInchi(Chem.MolFromSmiles(smi))

def query_fastsolv(grouped_dataset, solute_smiles, min_temp, max_temp):
    temperature_list = np.linspace(min_temp, max_temp, 20)
    solvent_list = []
    for solvent, row in grouped_dataset.loc[solute_smiles].iterrows():
        solvent_list.append(solvent)
    df_overall = pd.DataFrame()
    for temp in temperature_list:
        data = dict(
            solvent_smiles=solvent_list,
            solute_smiles=[solute_smiles]*len(solvent_list),
            temperature= [temp]*len(solvent_list), 
            )
        df = pd.DataFrame(data)
        df_overall = pd.concat([df, df_overall])
    return df_overall

def stats(true, predicted): 
    mse = mean_squared_error(true, predicted)

    differences = np.abs(true - predicted)
    within_1_unit = differences <= 1
    percentage_within_1_unit = np.sum(within_1_unit) / len(true) * 100
    return mse, percentage_within_1_unit

def parity_plot(xs,ys, labels, colors):
    plt.figure(figsize=[6.4,4.8])
    fig, ax1 = plt.subplots()
    ax1.plot([-6, 6], [-6,6], linestyle = '--', linewidth = 2, color = 'black')
    ax1.plot([-6, 6], [-7,5], linestyle = '--', linewidth = 2, color = 'gray', alpha = 0.4)
    ax1.plot([-6, 6], [-5,7], linestyle = '--', linewidth = 2, color = 'gray', alpha = 0.4)
    for i in range(len(xs)):
        ax1.scatter(xs[i],ys[i], alpha =0.4, s = 30, edgecolors = 'black', color = colors[i], label = labels[i])
    ax1.set_xlim([-5, 2])
    ax1.set_ylim([-5,2])
    ax1.set_ylabel(r"$\hat{logS}$")
    ax1.set_xlabel(r"True $logS$")
    ax1.legend(prop={'size': 20}, loc = 'upper left', frameon = False, bbox_to_anchor=(-0.05,1.1)) 
    ax1.spines[['right', 'top']].set_visible(False)
    return ax1

def solprop_parity_plot(x,y, label, color, mse, percentage_within_1_unit):
    plt.figure(figsize=[6.4,4.8])
    fig, ax1 = plt.subplots()
    ax1.plot([-6, 6], [-6,6], linestyle = '--', linewidth = 2, color = 'black')
    ax1.plot([-6, 6], [-7,5], linestyle = '--', linewidth = 2, color = 'gray', alpha = 0.4)
    ax1.plot([-6, 6], [-5,7], linestyle = '--', linewidth = 2, color = 'gray', alpha = 0.4)
    ax1.scatter(x,y, alpha =0.4, s = 30, edgecolors = 'black', color = color, label = label)
    ax1.set_xlim([-6,3])
    ax1.set_xticks([-6, -3, 0, 3])
    ax1.set_ylim([-6,3])
    ax1.set_yticks([-6, -3, 0, 3])
    ax1.set_ylabel(r"$\hat{logS}$")
    ax1.set_xlabel(r"True $logS$")
    ax1.legend(prop={'size': 20}, loc = 'upper left', frameon = False, bbox_to_anchor=(-0.05,1.1)) 
    ax1.text(-2,-4.5, f"RMSE = {np.sqrt(mse):.2f}", fontsize = 20)
    ax1.text(-2,-5.5, '% logS' + r'$\pm$'  + f"1 = {percentage_within_1_unit:.1f}", fontsize = 20)
    ax1.spines[['right', 'top']].set_visible(False)
    return ax1

def residual_cumsum(merged_fastprop, merged_chemprop, feature):
    
    #sort
    merged_fastprop_sorted_weight = merged_fastprop.sort_values(by=feature)
    merged_chemprop_sorted_weight = merged_chemprop.sort_values(by=feature)

    # Calculate cumulative residuals
    merged_fastprop_sorted_weight['cumulative_residual'] = merged_fastprop_sorted_weight['squared_residual'].cumsum()
    merged_chemprop_sorted_weight['cumulative_residual'] = merged_chemprop_sorted_weight['squared_residual'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(merged_fastprop_sorted_weight[feature], merged_fastprop_sorted_weight['cumulative_residual'], label='Fastprop', color='blue')
    plt.plot(merged_chemprop_sorted_weight[feature], merged_chemprop_sorted_weight['cumulative_residual'], label='Chemprop', color='red')
    plt.title('Cumulative Residuals vs ' + feature)
    plt.xlabel(feature)
    plt.ylabel('Cumulative Residual')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return
