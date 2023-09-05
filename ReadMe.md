# Beyond original Research Articles Categorization via NLP
### Rosanna Turrisi

[![Python 3.9.15](https://img.shields.io/badge/python-3.9.15-blue.svg)](https://www.python.org/downloads/release/python-392/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.11-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)


## ABSTRACT
This work proposes a novel approach for unsupervised text categorization
in the context of scientific literature, 
using Natural Language Processing techniques. 
The study leverages the power of pre-trained language models, specifically SciBERT, to extract meaningful representations of abstracts from the ArXiv dataset. Text categorization is performed using the K-Means algorithm, and the optimal number of clusters is determined based on the Silhouette score. The results demonstrate that the proposed approach 
captures subject information more effectively than the traditional arXiv labeling system, leading to improved text categorization. The approach offers potential for better navigation and recommendation systems in the rapidly growing landscape of scientific research literature.

## DATASET
The employed dataset is the open source [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). 

This is rich corpus of about 2 M articles, including publication information such as article titles, authors, subject categories, and abstracts. 
For computational reasons, a subset of articles published in 2023 was selected based on the journal-ref information. Additionally, certain filtering criteria were applied to ensure data quality.
Duplicated and withdrawn papers were removed from the dataset. Categories with a small number of papers, specifically those containing less than 250 articles, were also excluded. 

## FULL PAPER 
The full article is available in the Turrisi.pdf file.

## FOLDER CONTENT

- `run.sh` in which an example of how to run the whole pipeline in shown;


- `utilities.py`
This file contains useful functions related to categories visualisation, embedding creation and visualisation, clustering training and evaluation.
It also includes a dictionary that maps the arXiv category labels (e.g., 'astro-ph') into the corresponding subject name (e.g., 'Astrophysics'). 


- `data_analysis_and_processing.py`
This file selects a subset of the arXiv dataset (papers published after 2022), performs category labels and abstract lenght analysis, and abstract processing.


- `PCA_SciBert_emebedding.py`
This file extracts the last hidden layer of the pre-trained SciBert model and applies PCA method to obtain the final text representation.


- `FineTuned_SciBert_embedding.py`
This files performs fine-tuning of the pre-trained SciBert model on the subject category classification task. Finally, it extracts the last hidden layer of the fine-tuned model as text representation.


- `text_classification.py`
This file performs the unsupervised text classification based on K-Means algorithm by selecting the best number of classification categories based on Silhouette metric.
It also analyses the results on the testing set.


### GENERAL INFORMATION

All the images are saved in `data_path/img`

Fine-tuning results are saved in `data_path/finetuning`

All other results are saved in `data_path`

## Requirements
The codes have been tested on Python 3.8, Tensorflow 2.11.0, PyTorch 2.0.1. 
In order to run, the following Python modules       
are required:

- Numpy, SciPy, Scikit-learn, seaborn, json
- kneed
- os, sys, argparse, glob
- PyTorch, TensorFlow, transformers
- spacy, [en_core_sci_lg](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz)


