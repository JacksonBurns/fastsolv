---
title: "Predicting Non-Aqueous Solubility with Molecular Descriptors and Message Passing Neural Networks"
author: 
  - name: Jackson W. Burns \orcidlink{0000-0002-0657-9426}
    affil-id: 1,**
  - name: Lucas Attia \orcidlink{0000-0002-9941-3846}
    affil-id: 1,**
affiliations:
  - id: 1
    name: Massachusetts Institute of Technology, Cambridge, MA
  - id: "**"
    name: "Equal Contribution"
date: March 21, 2024
geometry: margin=1in
bibliography: paper.bib
citation-style: journal-of-cheminformatics
note: |
 This proposal can be compiled to other formats (pdf, word, etc.) using pandoc.

 To show the affiliations and ORCIDs, a special setup has been used. To recreate it locally,
 checkout this StackOverflow answer: https://stackoverflow.com/a/76630771 (this will require you
 to download a new default.latex and edit it as described in the post) _or_
 make sure that the version of pandoc you have installed is _exactly_ 3.1.6

 You can then compile this paper with:
   pandoc --citeproc -s proposal.md -o proposal.pdf --template default.latex
 from the proposal directory in fastprop.

 To compile _without_ doing all that annoying setup (so, on _any_ pandoc version),
 you can just leave off '--template default.latex' i.e.:
  pandoc --citeproc -s proposal.md -o proposal.pdf
 This won't render the author block correctly, but everything else should work fine.
note: |
 This paper has been copied from github.com/JacksonBurns/fastprop (and modified).
---

Notes from 4/3
- Look for other temperature dependent models
- Look for any models that have directly trained on Florence's dataset
- Consider other approaches to interact solute and solvent in a tabular/molecular descriptor feature rep.
  
The solubilities of drug-like molecules in non-aqueous organic solvents are crucial properties for drug substance and drug product manufacturing.
Experimentally measuring non-aqueous solid solubility requires notoriously tedious experiments which are both time-consuming and resource-intensive.
Thus, predicting organic solubility of drug-like molecules _a-priori_ based on their structure alone has been an active and robust area of academic and industrial research.
The traditional approach relies on empirical solubility models like the Abraham Solvation model to estimate solubility.
However, these empirical approaches are incapable of extrapolation by their nature, limited by the experimental data from which they are derived.
Recent work has instead turned to molecular Machine Learning (ML) which in theory could learn the underlying physics dictating the solubility and thus generalize.

Given the experimental challenges, datasets of organic solubility are highly dispersed in the literature.
Previous efforts by Vermeire et al. [@vermeire_solublility] compiled one of the largest publicly available datasets of thermodynamic quantities of drug-like molecules as well as a testing set of non-aqueous solubility for the same.
Using the former they trained a combination of three Directed-Message Passing (Graph) Neural Networks (D-MPNN) models (via Chemprop [@chemprop_theory; @chemprop_software]) to predict different thermodynamic quantities, which in turn predicted solubility using a thermocycle.
This coupling of an ML workflow to thermodynamics modeling achieved smooth gradients of solubility with respect to temperature.

While the reference model achieves a low error across most of chemical space, we have observed extremely poor performance and non-physical predictions in the limit of high solubility (>1 mol/L).
We believe this is due to the imbalanced nature of the training data, which has relatively few examples of highly soluble compounds.
By simplifying the learning task and instead directly predicting solubility from the input structures, we believe model performance can be improved.

We propose to train a Chemprop model which learns a representation for both the solute and solvent and then ingests the temperature by concatenating it to that representation.
Additionally, we will compare an alternative model architecture `fastprop`(GitHub.com/JacksonBurns/fastprop), which has been shown to outperform Chemprop on other prediction tasks using only classical molecular descriptors for the solute and solvent.
We will also quantify the performance of our models in the limit of high solubility to observe if direct prediction reduces the occurrence of non-physical predictions.
This approach could provide improved solubility predictions with greater interpretability, which could be a helpful contribution to solubility prediction.

# Data
We will be using the solubility dataset published by Vermeire et al. [@vermeire_solublility].
This dataset contains 6261 solubility datapoints, with solute and solvent SMILES, solubility (logS), and temperature (K) as features.

# Supervisor 
Given this is a molecular property prediction task on pharmaceutically-relevant small molecules, we would greatly appreciate Professor Coley's expertise on our project.

# Cited Works
