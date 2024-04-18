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

# Scientific Goal:
The solubilities of drug-like molecules in non-aqueous organic solvents are crucial properties for drug substance and drug product manufacturing.
Experimentally measuring non-aqueous solid solubility requires notoriously tedious experiments which are both time-consuming and resource-intensive.
Thus, predicting organic solubility of drug-like molecules _a-priori_ based on their structure alone has been an active and robust area of academic and industrial research.
The traditional approach relies on empirical solubility models like the Abraham Solvation model to estimate solubility.
However, these empirical approaches are incapable of extrapolation by their nature, limited by the experimental data from which they are derived.
We will extend recent work on applying molecular Machine Learning (ML) to this problem, which in theory could learn the underlying physics dictating the solubility and thus generalize.

# Problem Formulation and Background
This a supervised learning problem with many open questions related to molecular representation and overall model structure.
For the former, previous literature has focused primarily on applying learned representations via message-passing graph neural networks.
We will include this as a reference point but will instead primarily focus on the application of descriptor-based models via `fastprop` (GitHub.com/JacksonBurns/fastprop).
This comparison between representation approaches should prove to be informative.

For the latter point, prior literature has also usually tried to enforce physics constraints in the model architecture. 
Vermeire et al. [@vermeire_solublility] compiled one of the largest publicly available datasets of thermodynamic quantities of drug-like molecules as well as a testing set of non-aqueous solubility for the same.
Using the former they trained a combination of three Directed-Message Passing (Graph) Neural Networks (D-MPNN) models (via Chemprop [@chemprop_theory; @chemprop_software]) to predict different thermodynamic quantities, which in turn predicted solubility using a thermocycle.
Another work by Yashaswi and coauthors [@yashaswi_interaction] used an 'interaction block' - an intermediate layer in their network which performed a row-wise multiplication of the solute and solvent learned representations which was then passed to an FNN.
This approach is analogous to training the model to map the structures to abraham-like solubility parameters, which are then weighted and combined for prediction. In this project we will implement and compare various ways to interact the solvent and solute representations, including but not limited to simple concatenation, row-wise multiplication, dot product. The various ways to enforce interaction between the learned representations reflect different underlying functional forms of solvation physics, and to our knowledge, this would be the first work directly comparing different interaction architectures for the task of molecular organic solubility prediction. This question of appropriately enforcing physics in our model is likely the most interesting and challenging aspect of this project (see [Challenges](#challenges)). 

# Data
We will be using the aforementioned solubility dataset published by Vermeire et al. [@vermeire_solublility], which is made available via a machine-readable data format on Zenodo.
This dataset contains 6261 solubility datapoints, with solute and solvent SMILES, solubility (logS), and temperature (K) as features.
The original collators performed extensive data curation, so the reported solubility values are already well-sanitized and on a unified scale.
We may apply standard scaling, log scaling, or power scaling to the values to simplify prediction though this will ultimately be decided based on the performance.

# Challenges
We anticipate that the small amount of data and its highly imbalanced nature will require us to build physics in to our models.
The reference study which aggregated this data enforced physics by never directly training on the solubility and instead creating models to predict other molecular properties used to calculate it.
Our naive initial fastprop model will simply ingest the solute, solvent, and temperature as inputs to an FNN, effectively assuming that their is some ethereal non-linear mapping which can be learned between these and the solubility with no physics knowledge. The challenge is that their is likely some 'intermediate' between these two ideas which includes a sufficient amount of physics so as to assist the model in learning complex relationships but not so much that it becomes inflexible. By 'interacting' the solute and solvent representation (learned or descriptor-based) via element-wise multiplication, for example, we could force the model to learn a latent representation which is analogous to an abraham-like multiplicative solubility coefficient. Finding which 'interaction' between these representations is the most effective will require creativity and extensive experimentation.

# Cited Works
