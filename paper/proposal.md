---
title: "Generalizable, Fast, and Accurate Deep-QSPR with `fastprop`"
subtitle: "Part 1: Framework and Benchmarks"
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
# Direct nonaqeuous solubility prediction from mordred descriptors and message passing neural networks

The solubilities of drug-like molecules in nonaqeuous organic solvents are crucial properties for drug substance and drug product manufacturing. Experimentally predicting nonaqeuous solid solubility requires notoriously tedious experiments which are time-consuming and can be resource-intensive. Thus, predicting organic solubility of drug-like molecules has been an active and robust area of academic and industrial research. The difficulty of these experiments has traditionally generated a reliance on empirical solubility models, like the Abraham Solvation model, to get an estimate of the solubility. However, these empirical approaches are also limited by the experimental data from which they are derived. Thus, much recent work has turned to molecular machine learning (ML) to predict solubility, which in theory could learn the underlying physics dictating the solubility and generalize. However, given the experimental challenges, datasets of organic solubility are highly dispersed in the literature. Previous efforts by Vermeire et al. [@vermeire_solublility] compiled one of the largest publically available datasets of nonaqueous solubility of drug-like molecules. Using this dataset, the trained a combination of three D-MPNN models (via Chemprop [@chemprop_theory; @chemprop_software]) to directly predict different thermodynamic quantities, which in turn predicted solubility using a thermocycle. This coupling of an ML workflow to thermodynamics modeling proved to acheive smooth gradients of solubility with respect to temperature. While this model performs reasonably well, we have observed extremely poor performance and non-physical predictions in the limit of high solubility. Thus, we hypothesized that directly training a model to predict solubility on the same dataset can improve model performance, since the learning task is simpler. Thus, we propose to instead train a Chemprop model which takes the temperature as an extra feature concatenated to the solvent and solute learned representations. Additionally, we will compare an alternative model architecture `fastprop`(GitHub.com/JacksonBurns/fastprop), which has been shown to outperform Chemprop on other prediction tasks using only classical molecular descriptors for the solute and solvent. We will also quantify the performance of our models in the limit of high solubility to observe if direct prediction reduces the occurrence of non-physical predictions. This approach could provide improved solubility predictions with greater interpretability, which could be a helpful contribution to solubility prediction. 

# Data
We will be using the solubility dataset published by Vermeire et al. [@vermeire_solublility]. This dataset contains 6261 solubility datapoints, with solute and solvent SMILES, solubility (logS), and temperature (K) as features. 


# Supervisor 
Given this is a molecular property prediction task on pharmaceutically-relevant small molecules, we would greatly appreciate Professor Coley's expertise on our project. 

# Cited Works
