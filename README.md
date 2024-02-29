# highsol

Story:
 - Vermeire devised a method to predict solubility of arbitrary combinations of molecules at any temperature.
 - Attia was using this for an actual system in research, it was supposed to be highly soluble, the model predicted a non-physical high value.
 - On further investigation, _approximately_ 90% of the data falls in the range of insoluble to 1/3 of the max, 10% in the range of 1/3rd most to 2/3rd most, and only 1% of the dataset is in the range of very high solubility the remaining 1/3rd of the solublity range
 - Ideas: apply power scaling to smooth this range, then train another chemprop model to see if it works; train a fastprop model on the whole data and this range for comparison.
 - Limitation - not interested in any other than STP, so retraining the model in the future will be required for fair comparison (unless we want to add temperature to fastprop, which is possible) after we use the published model just to check the accuracy on this limited range

Steps:
 1. Identify a formal metric (what constitutes highly soluble + an error measurement) to quantify the perceived 'issue' at the limit of high solubility; evaluate the published Chemprop model on the subset of highly soluble data which is identified.
 2. We will __not__ try to improve the published Chemprop model, but could:
    - Power scale the entire dataset, retrain.
    - Retrain on _just_ the highly soluble data.
    - Recast the problem as classification, train.
 3. Train an alternative architecture like `fastprop` on the whole dataset and the highly soluble subset.
 4. Train a 'plain' Chemprop model on just the highly soluble data.
    - Reducing model complexity will be required when reducing dataset size, so rather than re-training the original '3 models in a trenchcoat' Florence model which is very data hungry due to the large number of parameters, we will instead train a new Chemprop model that directly predicts solubility given pairs of molecules and the temperature (as an extra feature to the FNN)
    Additional reason we are doing this is because training the Florence model again is very hard.
    - 

Florence model supports arbitrary combinations of molecules at an arbitrary temperature, uses a complicated thermocycle to predict properties for adjusting baseline estimates.
Alternative approach is to treat temperature as an extra feature and just learn on combinations.

The dataset is here: https://zenodo.org/records/5970538
The solubility data is separated into two CSV files: `CombiSolu-Exp.csv` and `CombiSolu-Exp-HighT.csv` located in `SolProp_v1.2/Data`.
The former contains the embeddings for the two molecules, the temperature at which the solubility measurement was made, as well as the actual solubility measurement in mol fraction (converted from various literature reporting formats).
The latter contains the same data as the former, but at higher temperatures.

Reference model has data leaks:
 - solvents and solutes are not split separately for _any_ of the three models, meaning inference time will have already seen at least one of the two species
 - temperatures are also not stratified, so it is possible during inference time that the model will have already seen that combination exactly
 - adjusting the solubility for temperature based on $S_aq$ leaks data for solvents as well

We can have two splitting approaches:
 - ignore this, and randomly split data as done in the reference study (RMSE/MAE around 1-2/0.5-1)
 - no temperature should appear in all splits, no molecules (either solvent or solute) should appear in all splits - to deal with this (1) generate a list of all molecules (2) split _that_ list (3) generate all pairs of molecules in the training split (4) assemble the actual training data by looking up all of those combinations that are actually in the dataset (as long as the data is evenly sparse, this should be ok).

**Solubility is log-transformed (base 10)**
