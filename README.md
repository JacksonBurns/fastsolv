# highsol

# Prediction of Solubility for Highly SOluble Compounds

Story:
 - Vermeire devised a method to predict solubility of arbitrary combinations of molecules at any temperature.
 - Attia was using this for an actual system in research, it was supposed to be highly soluble, the model predicted a non-physical high value.
 - On further investigation, _approximately_ 90% of the data falls in the range of insoluble to 1/3 of the max, 10% in the range of 1/3rd most to 2/3rd most, and only 1% of the dataset is in the range of very high solubility the remaining 1/3rd of the solublity range
 - Ideas: apply power scaling to smooth this range, then train another chemprop model to see if it works; train a fastprop model on the whole data and this range for comparison.
 - Limitation - not interested in any other than STP, so retraining the model in the future will be required for fair comparison (unless we want to add temperature to fastprop, which is possible) after we use the published model just to check the accuracy on this limited range
