# highsol
This repository contains the code and paper for `Predicting Non-Aqueous Solubility with Molecular Descriptors and Message Passing Neural Networks` by Burns, Attia, Doyle, and Green.

The layout of the repository is as follows:
 - `models`: code for running the models referenced in the paper.
 - `paper`: source files for the corresponding paper.
 - `data`: data munging scripts for preparing the source data to work with `fastprop` and Chemprop.



# TODO
 - remove test set from krasnov training, train on 95:5 training/validation split
 - drop solutes from leeds that are in Krasnov (arleady done)
 - drop solutes from krasnov that are in solprop
These two are justifiable by the numer of solutes in each - want to preserve solute diversity

remove temperature skip connection

Re-optimize models with smaller Krasnov set
