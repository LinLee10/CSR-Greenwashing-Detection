<h1 align="center">Greenwashing in Corporate Sustainability Statements</h1>

<p align="center"><b>Best test accuracy 0.99</b> · <b>Baseline accuracy 0.52</b> · <b>500 labeled statements</b></p>

## Overview

This project is an NLP research study from the UC Berkeley School of Information on detecting greenwashing in corporate sustainability statements.

Given a short passage from a corporate or CSR report, the models classify it as one of three labels:

* Genuine
* Mixed
* Greenwashing

I treat this as both a three class classification task and an ordered prediction task, moving from simple baselines to a fine tuned transformer model with clean experimental design and clear metrics.

## Dataset and labels

The dataset in `Data.csv` contains:

* 500 statements from corporate sustainability and CSR materials
* One text column `original_text`
* One numeric label column `label` where 1 means Genuine, 2 means Mixed, 3 means Greenwashing
* Class imbalance where Genuine is most common, Mixed is moderate, Greenwashing is rare

I model the task as three class classification and as ordered classification along a greenwashing severity scale where Genuine is less severe than Mixed and Mixed is less severe than Greenwashing.

## Models and key results

All models share the same setup:

* Single shuffle of the data
* Train dev test split with proportions 60 20 20
* Training on the train set, model selection on the dev set, final reporting on the test set

Test accuracy on the held out test set:

* Majority baseline (always predicts Genuine)  accuracy about 0.52
* TFIDF logistic regression accuracy about 0.92
* TFIDF ordinal logistic regression accuracy about 0.95
* Fine tuned BERT classifier accuracy about 0.99

Across models I compute accuracy, confusion matrices, and simple confidence intervals so that the gains from classical models to the transformer model are statistically meaningful, not just numerical noise.

## Running the project

* Open `Model Training Code.ipynb` in Jupyter or another notebook environment
* Ensure standard Python libraries for data science and transformers are installed
* Keep `Data.csv` in the same folder as the notebook
* Run the cells from top to bottom to reproduce data loading, training, and evaluation
