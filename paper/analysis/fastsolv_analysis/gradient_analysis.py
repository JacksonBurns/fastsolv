import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

def gradient_analysis(df):
    grouped_predictions = df.groupby(['solvent_smiles', 'solute_smiles'])
    # Create a list of smaller dataframes
    sub_dfs = [group for _, group in grouped_predictions]

    total_predicted_gradients = []
    total_true_gradients = []
    for df in sub_dfs:
        if(len(df) > 1):
            total_true_gradients.append(np.gradient(df['logS_true'], df['temperature']).flatten())
            total_predicted_gradients.append(np.gradient(df['logS_pred'], df['temperature']).flatten())
    
    true_grads = np.concatenate(total_true_gradients).ravel()
    pred_grads = np.concatenate(total_predicted_gradients).ravel()
    mask = np.isfinite(true_grads) & np.isfinite(pred_grads) 
    true_grads = true_grads[mask]
    pred_grads = pred_grads[mask]
    
    mse = mean_squared_error(true_grads, pred_grads)
    mae = mean_absolute_error(true_grads, pred_grads)

    return true_grads, pred_grads, mse, mae

def gradient_parity_plot(true_grads, pred_grads):
    plt.figure(figsize=[6.4,4.8])
    fig, ax1 = plt.subplots()

    bins = [100, 100] # number of bins
    hh, locx, locy = np.histogram2d(true_grads, pred_grads, bins=bins, density = True)

    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(true_grads, pred_grads)])
    idx = z.argsort()
    x2, y2, z2 = true_grads[idx], pred_grads[idx], z[idx]


    ax1.scatter(true_grads,pred_grads, c = z2, alpha =0.4, edgecolors = 'black')
    ax1.plot([-1, 1], [-1,1], linestyle = '-', linewidth = 3, color = 'black')

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax1)
    cbar.ax.set_ylabel('Density')


    ax1.set_xlim([-0.1, 0.1])
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_ylabel(r"$\frac{\hat{dlogS}}{dT}$")
    ax1.set_xlabel(r"$\frac{dlogS}{dT}$")
    ax1.legend(prop={'size': 22}, loc = 'upper left', frameon = False) 
    ax1.spines[['right', 'top']].set_visible(False)
    return ax1

def pdf_plot(true_grads, pred_grads, bins):
    true_grads_binned, true_grad_bins = np.histogram(true_grads, bins = bins, range = (-0.05, 0.05))
    pred_grads_binned, pred_grad_bins = np.histogram(pred_grads, bins = bins, range = (-0.05, 0.05))

    plt.figure(figsize=[6.4,4.8])
    ax1 = plt.gca()

    ax1.stairs(true_grads_binned, true_grad_bins, label = r'ground truth')
    ax1.stairs(pred_grads_binned,pred_grad_bins, label = r'predictions')
    #ax1.set_xlim([-0.05, 0.05])
    #ax1.set_ylim([-500, 22000])
    ax1.set_ylabel(r"Count")
    ax1.set_xlabel(r"$\frac{dlogS}{dT}$")
    ax1.legend(prop={'size': 22}, loc = 'upper right', frameon = False) 
    ax1.spines[['right', 'top']].set_visible(False)
    return ax1, true_grads_binned, pred_grads_binned, true_grad_bins, pred_grad_bins

def cdf_plot(true_grads, pred_grads, bins): 
    # Compute histograms
    true_grads_binned, true_grad_bins = np.histogram(true_grads, bins=bins, range = (-0.05, 0.05))
    pred_grads_binned, pred_grad_bins = np.histogram(pred_grads, bins=bins, range = (-0.05, 0.05))

    # Compute the CDF
    true_grads_cdf = np.cumsum(true_grads_binned) / np.sum(true_grads_binned)
    pred_grads_cdf = np.cumsum(pred_grads_binned) / np.sum(pred_grads_binned)

    distance = wasserstein_distance(true_grads_cdf, pred_grads_cdf)

    # Create the plot
    plt.figure(figsize=[6.4, 4.8])
    ax1 = plt.gca()

    # Plot the CDFs as step plots
    ax1.step(true_grad_bins[:-1], true_grads_cdf, where='post', label='ground truth')
    ax1.step(pred_grad_bins[:-1], pred_grads_cdf, where='post', label='predictions')   
    ax1.text(0.02, 0.2, f"EMD = {distance:.2f}", color = 'orange')
    # Set axis limits
    ax1.set_ylim([0, 1])
    ax1.set_yticks([0,0.5 ,1])

    ax1.set_ylabel("Cumulative Probability")
    ax1.set_xlabel(r"$\frac{dlogS}{dT}$")
    ax1.legend(prop={'size': 22}, loc='upper left', frameon=False)
    ax1.spines[['right', 'top']].set_visible(False)
    return ax1, true_grads_cdf, pred_grads_cdf, distance, true_grad_bins, pred_grad_bins


def parity_plot(x,y, label, color, mse, percentage_within_1_unit):
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