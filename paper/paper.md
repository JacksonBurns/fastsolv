---
title: "Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`"
subtitle: "Part 1: Framework and Benchmarks"
author: 
  - name: Jackson W. Burns \orcidlink{0000-0002-0657-9426}
    affil-id: 1,**
  - name: Lucas Attia \orcidlink{0000-0002-9941-3846}
    affil-id: 1,**
  - name: Patrick S. Doyle \orcidlink{0000-0003-2147-9172}
    affil-id: 1
  - name: William H. Green \orcidlink{0000-0003-2603-9694}
    affil-id: 1,*
affiliations:
  - id: 1
    name: Massachusetts Institute of Technology, Cambridge, MA
  - id: "*"
    name: "Corresponding: whgreen@mit.edu"
  - id: "**"
    name: "Equal Contribution"
date: March 12, 2024
geometry: margin=1in
bibliography: paper.bib
citation-style: journal-of-cheminformatics
note: |
 This paper can be compiled to other formats (pdf, word, etc.) using pandoc.

 To show the affiliations and ORCIDs, a special setup has been used. To recreate it locally,
 checkout this StackOverflow answer: https://stackoverflow.com/a/76630771 (this will require you
 to download a new default.latex and edit it as described in the post) _or_
 make sure that the version of pandoc you have installed is _exactly_ 3.1.6

 You can then compile this paper with:
   pandoc --citeproc -s paper.md -o paper.pdf --template default.latex
 from the paper directory in fastprop.

 To compile _without_ doing all that annoying setup (so, on _any_ pandoc version),
 you can just leave off '--template default.latex' i.e.:
  pandoc --citeproc -s paper.md -o paper.pdf
 This won't render the author block correctly, but everything else should work fine.
note: |
 This paper has been copied from github.com/JacksonBurns/fastprop (and modified).
---

# Summary
 1. Experimentalists find it useful to know the solubility of a solid in a solvent _a-priori_ to aid with synthesis.
 2. Prior experimental data with known solubilities is both highly dispersed in the literature and incomplete.
 3. This forces experimentalists to rely on models like the Abraham Solvation model to get an estimate of the solubility (which is also limited by the experimental data from which the model was derived) or as a lost resort simple empirical work which is time consuming.
 4. Predicting solubility for _arbitrary_ pairs of solid solutes and liquid solvents at _arbitrary_ temperatures was then of course a perfect setting for Artificial Intelligence (AI), which in theory could learn the underlying physics dictating the solubility and make predictions outside of its experimental training.
 5. Previous efforts by Vermeire et al. [@vermeire_solublility] used a combination of three D-MPNN models (via Chemprop [@chemprop_theory; @chemprop_software]) to predict the solubility: one which directly predicted solubility at room temperature and two others which predicted other thermodynamic properties needed to complete a thermocycle to adjust that initial prediction to the desired temperature. The best-case performance from that study was an RMSE of 0.44 and an MAE of 0.29.
 6. We have demonstrated that we can instead train a model which takes the temperature as an extra feature concatenated to the solvent and solute learned representations that outperforms the reference study after hyperparameter optimization. Performance metrics: ... 
 7. Even more noteworthy is that an alternative model architecture `fastprop` (GitHub.com/JacksonBurns/fastprop) is able to outperform both of these models using only classical molecular descriptors for the solute and solvent, again with the temperature concatenated. Performance metrics: ... (include (W)MAPE).
 8. Separately from the pure improvement of the model is the interest in the 'high solubility' domain. It is often most interesting to identify which solvents are the _best_ at dissolving a given solid, so it is critical for the models to be especially accurate in this region of the data (are they?).

# Results and Discussion

Out of the box `fastprop` results:
```
[03/12/2024 12:56:40 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    4.0  0.026871  0.004248  0.020674  0.025870  0.028423  0.029424  0.029964
validation_r2          4.0  0.972656  0.004590  0.967403  0.970829  0.972319  0.974147  0.978583
validation_mape        4.0  0.453042  0.385711  0.210034  0.242433  0.287313  0.497922  1.027508
validation_wmape       4.0  0.076513  0.008897  0.067193  0.070593  0.075767  0.081687  0.087323
validation_l1          4.0  0.134489  0.020476  0.117165  0.123618  0.128406  0.139277  0.163981
validation_mdae        4.0  0.080012  0.023635  0.066624  0.067542  0.069015  0.081485  0.115395
validation_rmse        4.0  0.233760  0.018872  0.206196  0.229746  0.240361  0.244375  0.248122
[03/12/2024 12:56:40 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    4.0  0.027580  0.015438  0.015501  0.019754  0.022316  0.030143  0.050187
test_r2          4.0  0.972353  0.014206  0.951648  0.969449  0.977098  0.980003  0.983569
test_mape        4.0  0.445411  0.243454  0.261824  0.312939  0.358724  0.491197  0.802375
test_wmape       4.0  0.075826  0.016898  0.064619  0.067681  0.068844  0.076988  0.100996
test_l1          4.0  0.133043  0.024908  0.114600  0.118541  0.124067  0.138569  0.169437
test_mdae        4.0  0.077020  0.017380  0.063228  0.067340  0.071297  0.080978  0.102260
test_rmse        4.0  0.231165  0.060701  0.178544  0.201060  0.213808  0.243913  0.318499
[03/12/2024 12:56:40 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.938>0.05.
[03/12/2024 12:56:40 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[03/12/2024 12:56:40 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:05:58.338315
```

Out of the box Chemprop results:
```
Moving model to cuda
Model 0 test rmse = 0.525936                                                                                                                                                                                                                                      
Model 0 test mae = 0.389645
Model 0 test r2 = 0.882762
Ensemble test rmse = 0.525936
Ensemble test mae = 0.389645
Ensemble test r2 = 0.882762
1-fold cross validation
        Seed 0 ==> test rmse = 0.525936
        Seed 0 ==> test mae = 0.389645
        Seed 0 ==> test r2 = 0.882762
Overall test rmse = 0.525936 +/- 0.000000
Overall test mae = 0.389645 +/- 0.000000
Overall test r2 = 0.882762 +/- 0.000000
Elapsed time = 0:04:37
```

If we drop all datapoints where the solubility is less than 1 mol/L, performance changes dramatically.

For `fastprop`:
```
[03/12/2024 01:23:35 PM fastprop.fastprop_core] INFO: Displaying validation results:
                     count      mean       std       min       25%       50%       75%       max
validation_mse_loss    4.0  0.257756  0.094608  0.151753  0.192381  0.263613  0.328988  0.352045
validation_r2          4.0  0.714407  0.127950  0.599541  0.605701  0.715413  0.824119  0.827263
validation_mape        4.0  1.281281  0.630359  0.550676  0.859609  1.356428  1.778099  1.861591
validation_wmape       4.0  0.257741  0.067712  0.192016  0.215704  0.245380  0.287417  0.348188
validation_l1          4.0  0.116890  0.028396  0.081056  0.103550  0.119001  0.132341  0.148505
validation_mdae        4.0  0.083636  0.034056  0.043839  0.069830  0.081935  0.095741  0.126836
validation_rmse        4.0  0.166500  0.032278  0.130036  0.144165  0.169537  0.191872  0.196892
[03/12/2024 01:23:35 PM fastprop.fastprop_core] INFO: Displaying testing results:
               count      mean       std       min       25%       50%       75%       max
test_mse_loss    4.0  0.265054  0.066032  0.182714  0.227933  0.273366  0.310487  0.330772
test_r2          4.0  0.708471  0.075085  0.615578  0.668376  0.714235  0.754330  0.789837
test_mape        4.0  2.460535  1.070645  1.086656  1.975646  2.564792  3.049682  3.625901
test_wmape       4.0  0.250249  0.037491  0.195151  0.244520  0.263510  0.269238  0.278824
test_l1          4.0  0.116859  0.017160  0.091285  0.115636  0.124012  0.125235  0.128125
test_mdae        4.0  0.081192  0.018433  0.057189  0.072727  0.083629  0.092093  0.100319
test_rmse        4.0  0.170068  0.022425  0.142686  0.156965  0.172303  0.185406  0.192982
[03/12/2024 01:23:35 PM fastprop.fastprop_core] INFO: 2-sided T-test between validation and testing rmse yielded p value of p=0.862>0.05.
[03/12/2024 01:23:35 PM fastprop.cli.fastprop_cli] INFO: If you use fastprop in published work, please cite: ...WIP...
[03/12/2024 01:23:35 PM fastprop.cli.fastprop_cli] INFO: Total elapsed time: 0:01:48.665913
```

and for chemprop:
```
Moving model to cuda
Model 0 test rmse = 0.266730                                                                                                                                                                                                                                      
Model 0 test mae = 0.202236
Model 0 test r2 = 0.329536
Ensemble test rmse = 0.266730
Ensemble test mae = 0.202236
Ensemble test r2 = 0.329536
1-fold cross validation
        Seed 0 ==> test rmse = 0.266730
        Seed 0 ==> test mae = 0.202236
        Seed 0 ==> test r2 = 0.329536
Overall test rmse = 0.266730 +/- 0.000000
Overall test mae = 0.202236 +/- 0.000000
Overall test r2 = 0.329536 +/- 0.000000
Elapsed time = 0:00:49
```

# Declarations

## Availability of data and materials
All associated code for this paper can be accessed on GitHub at GitHub.com/JacksonBurns/highsol.

All data used are available to the public under a permissive license.
See the GitHub repository for more information on retrieving the data.

## Funding
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0023112.

## Acknowledgements
Yes, perhaps, indeed.

## Disclaimer
This report was prepared as an account of work sponsored by an agency of the United States Government.
Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States Government or any agency
thereof.
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# Cited Works