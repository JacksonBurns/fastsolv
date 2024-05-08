---
title: "Predicting Non-Aqueous Solubility with Molecular Descriptors and Message Passing Neural Networks"
author: 
  - name: Lucas Attia \orcidlink{0000-0002-9941-3846}
    affil-id: 1,**
  - name: Jackson W. Burns \orcidlink{0000-0002-0657-9426}
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

# Introduction and Related Work
The solubilities of drug-like molecules in non-aqueous organic solvents are crucial properties for drug substance and drug product manufacturing.[@hewitt2009silico] 
Experimentally measuring non-aqueous solid solubility requires notoriously tedious experiments which are both time-consuming and resource-intensive.[@alsenz2007high]
Thus, predicting organic solubility of drug-like molecules _a-priori_ based on their structure alone has been an active and robust area of academic and industrial research.[@jorgensen2002prediction]
The traditional approach for prediction relies on empirical solubility models like the Abraham Solvation model to estimate solubility.[@taft1985linear]
However, these empirical approaches are incapable of extrapolation by their nature, limited by the experimental data from which they are derived. Recent work has instead explored applying molecular Machine Learning (ML) to this problem, which in theory could learn the underlying physics dictating the solubility and thus generalize to new solutes, solvents, and temperatures.[@lusci2013deep;@panapitiya2022evaluation] Numerous approaches have been developed, leveraging deep neural networks (NNs) learning from molecular fingerprints,[@zang2017silico] molecular descriptors,[@boobier2020machine;tayyebi2023prediction] graph convolutional NNs (GCNNs),[@chemprop_theory], among other architectures. Extensive work has compared the performance of models based on molecular graph representations and models based on molecular descriptors. Here, we focus on leveraging a recently developed model architecture `fastprop` (GitHub.com/JacksonBurns/fastprop), which has been demonstrated to outperform D-MPNN-based models in several molecular property prediction tasks.[jackson add citation to his paper] Fastprop is a descriptor-based model, which uses Mordred descriptors[@moriwaki2018mordred] in a simple feedforward NN (FNN). 

In parallel, there has been an effort to integrate the physics of solvation into the ML model architecture. For example, previous efforts by Vermeire et al. [@vermeire_solublility] combined three directed-message passing NN (D-MPNN) models (via Chemprop [@chemprop_theory; @chemprop_software]) trained to predict different thermodynamic quantities to predict solubility. While this model demonstrates impressive performance in ultimately predicting solubility, the approach inherently leaks data, since all solvent and solute molecules are seen during training. A stringent data splitting approach would likely reveal worse model performance. Another work by Yashaswi and coauthors [@yashaswi_interaction] used an 'interaction block' - an intermediate layer in their network which performed a row-wise multiplication of the solute and solvent learned representations which was then passed to an FNN. This approach is analogous to training the model to map the structures to abraham-like solubility parameters, which are then weighted and combined for prediction. 
Here, we hypothesized that:
  (1) Directly training a fastprop-based model on the dataset curated by Vermeire et al. could lead to better solubility prediction than a D-MPNN-based model 
  (2) Incorporating physics of solvation into the interaction between solute and solvent interactions in a fastprop-based model could further improve solubility predictions. 

To this end, we first optimize naive fastprop and chemprop models on the Vermeire dataset. We then present a physics-infused fastprop architecture, which has branches that learn solute and solvent representations, then combine these representations in specific computations in an interaction block. Specifically, we implement and compare various ways to interact the solvent and solute representations, including simple concatenation, row-wise multiplication, dot product. The various ways to enforce interaction between the learned representations reflect different underlying functional forms of solvation physics, and to our knowledge, this would be the first work directly comparing different interaction architectures for the task of molecular organic solubility prediction. This question of appropriately enforcing physics in our model is likely the most interesting and challenging aspect of this project. We optimize a model using this architecture, then compare to the optimized model baselines. Finally, since it is difficult to compare model performance directly to Vermeire et al. due to the previously specified data leak, we compared our model performance against datasets compiled by Boobier et al., which directly predicts solubility.[@boobier2020machine]

## Data
We use the aforementioned solubility dataset published by Vermeire et al. [@vermeire_solublility], which is made available via a machine-readable data format on Zenodo. This dataset contains 6261 solubility datapoints, with solute and solvent SMILES, solubility (logS), and temperature (K) as features. The original collators performed extensive data curation, so the reported solubility values are already well-sanitized and on a unified scale. We apply standard scaling, log scaling, or power scaling to the values to simplify prediction though this will ultimately be decided based on the performance.

The dataset is here: https://zenodo.org/records/5970538
The solubility data is separated into two CSV files: `CombiSolu-Exp.csv` and `CombiSolu-Exp-HighT.csv` located in `SolProp_v1.2/Data`.
The former contains the embeddings for the two molecules, the temperature at which the solubility measurement was made, as well as the actual solubility measurement in mol fraction (converted from various literature reporting formats).
The latter contains the same data as the former, but at higher temperatures.

**Solubility is log-transformed (base 10)**

# Methods
We anticipate that the small amount of data and its highly imbalanced nature will require us to build physics in to our models.
The reference study which aggregated this data enforced physics by never directly training on the solubility and instead creating models to predict other molecular properties used to calculate it.
Our naive initial fastprop model will simply ingest the solute, solvent, and temperature as inputs to an FNN, effectively assuming that their is some ethereal non-linear mapping which can be learned between these and the solubility with no physics knowledge. The challenge is that their is likely some 'intermediate' between these two ideas which includes a sufficient amount of physics so as to assist the model in learning complex relationships but not so much that it becomes inflexible. By 'interacting' the solute and solvent representation (learned or descriptor-based) via element-wise multiplication, for example, we could force the model to learn a latent representation which is analogous to an abraham-like multiplicative solubility coefficient. Finding which 'interaction' between these representations is the most effective will require creativity and extensive experimentation.

There are a ton of methods to predict solubility of organics in water (and probably common solvents, like octanol: https://www.sciencedirect.com/science/article/abs/pii/S002235491531025X) -> there are fewer methods to predict in arbitrary solvent -> even fewer to predict in arbitrary solvent at arbitrary temperature
Group contribution methods are abundant for the first one
second one: Common approach is fitting to experimental data, like the Abraham Solvation model or this paper: https://www.sciencedirect.com/science/article/abs/pii/S0022354915327301 which builds on that model
third one seems to be basically only done with ML, see this predecessor to Vermeire's work: https://link.springer.com/article/10.1186/s13321-021-00575-3 though I did find one interesting paper that does it with UniFac (!!) https://pubs.acs.org/doi/full/10.1021/ie011014w
I have thought more about interaction blocks and have grown very skeptical. Here's my thought process:
we have two representations for two molecules (learned or descriptors) that we know are physically interacting with eachother.
We want the FNN to learn this interaction, and specifically how it correlates to the solubility.
The interaction between the features could be any linear or nonlinear function.
Key: there is no mathematical interaction we can enforce that the network would be unable to learn on its own since NNs are universal function approximators
example: if the 'correct' (heavy quotes) interaction between the solute and the solvent was multiplication of each feature by the corresponding one in the other molecule, the NN could readily learn this. There is no point in doing this manually, except to perhaps make the task easier for the network. Critically, existing methods actually make the task harder by keeping the un-interacted representations, making the total parameter space larger.
Using an 'interaction block' only increases the parameter space unless you actually replace the features with the interaction, which we know we shouldn't do because no mathematical operation we can do would include non-linear interactions.
I propose that we try some interaction blocks, if only to prove that they just make the task harder, and the network just ends up learning based on the un-interacted input.
We can do this by looking at the feature importance's of the inputs of the FNN or by just comparing the performance values.

![`fastprop` logo.\label{logo}](../fastprop_logo.png){ width=2in }

# Results
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

Optimized Chemprop results:
```Best trial, with seed 164
{'depth': 6, 'dropout': 0.0, 'ffn_num_layers': 3, 'linked_hidden_size': 1400}
num params: 14,513,801
Elapsed time = 19:41:28

Overall test rmse = 0.373272 +/- 0.000000
Overall test mae = 0.279181 +/- 0.000000
Overall test r2 = 0.940946 +/- 0.000000
Elapsed time = 0:10:31
```

and with repetitions:
```
Overall test rmse = 0.373272 +/- 0.000000
Overall test mae = 0.279181 +/- 0.000000
Overall test r2 = 0.940946 +/- 0.000000
Elapsed time = 0:10:42

Overall test rmse = 0.375874 +/- 0.000000
Overall test mae = 0.270744 +/- 0.000000
Overall test r2 = 0.937207 +/- 0.000000
Elapsed time = 0:10:39

Overall test rmse = 0.398474 +/- 0.000000
Overall test mae = 0.294095 +/- 0.000000
Overall test r2 = 0.922584 +/- 0.000000
Elapsed time = 0:10:36

Overall test rmse = 0.446716 +/- 0.000000
Overall test mae = 0.330963 +/- 0.000000
Overall test r2 = 0.908125 +/- 0.000000
Elapsed time = 0:10:37
```

<!-- Consider adding this section about highly soluble molecules back to the paper for submission to a journal - it could prove interesting as a comment on 'hit detection', an interesting application of these models.

## Highly Soluble Species
Story:
 - Vermeire devised a method to predict solubility of arbitrary combinations of molecules at any temperature.
 - Attia was using this for an actual system in research, it was supposed to be highly soluble, the model predicted a non-physical high value.
 - On further investigation, _approximately_ 90% of the data falls in the range of insoluble to 1/3 of the max, 10% in the range of 1/3rd most to 2/3rd most, and only 1% of the dataset is in the range of very high solubility the remaining 1/3rd of the solublity range
 - Ideas: apply power scaling to smooth this range, then train another chemprop model to see if it works; train a fastprop model on the whole data and this range for comparison.
 - Limitation - not interested in any other than STP, so retraining the model in the future will be required for fair comparison (unless we want to add temperature to fastprop, which is possible) after we use the published model just to check the accuracy on this limited range 

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
-->

# Conclusion


<!-- These two sections can be removed after submitting the class report - they are likely not needed for a journal submission. -->
# Contributions

# Code
Code is available via GitHub

# Results and Discussion

<!-- These sections should be added back for the eventual paper submission.
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
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof. -->

# Cited Works
