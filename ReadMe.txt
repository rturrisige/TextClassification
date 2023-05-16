			TEXT CLASSIFICATION INTO AN UNKNOWN NUMBER OF CATEGORIES

# OBJECTIVE
This study aims to classify research articles, based on their abstract, into 𝑁 meaningful categories. Subject category labels are provided, but the purpose of this project is to go beyond the original labelling to find a better category representation. The obtained classes should of course contain information about the original subject categories. The optimal number of categories 𝑁 is unknown and finding its best value is one of the main challenges of the project. To this end, a Natural Language Processing (NLP) pipeline is developed by using Python language. 

# DATASET
The employed dataset is the open source arXiv Dataset (available at https://www.kaggle.com/datasets/Cornell-University/arxiv). This is rich corpus of about 2 M articles, including publication information such as article titles, authors, subject categories, and abstracts. For computational reasons, in this project only a subset of the corpus has been used

# FOLDER CONTENT

- requirements.txt in which the necessary Python package are reported;
- run.sh in which an example of how to run the whole pipeline in shown;
- Python files performing the NLP pipeline.


## PYTHON FILES

--------- utilities.py ----------
This file contains useful functions related to categories visualisation, embedding creation and visualisation, clustering training and evaluation.
It also includes a dictionary that maps the arXiv category labels (e.g., 'astro-ph') into the corresponding subject name (e.g., 'Astrophysics'). 

- data_analysis_and_processing.py
This file selects a subset of the arXiv dataset (papers published after 2022), performs category labels and abstract lenght analysis, and abstract processing. 

INPUT
data_path, i.e. the path to the folder that contains the arXiv dataset "arxiv-metadata-oai-snapshot.json". 

OUTPUT
categories_labels.npy: the arXiv category labels of the selected subset.
Numpy array of dimension n_sample x n_categories. Saved in data_path.
 
unique_categories_dictionary_from2022.npy: a dictionary mapping the subject category name into the label.
Numpy dictionary with n_categories elements. Saved in data_path.

abstract_length_statistics_data_from2022.csv: statistic description of the abstracts lenght.
Pandas data frame. Saved in data_path.

preprocessed_data_from2022.csv: dataset with pre-processed abstracts and categories labels.
Pandas data frame of dimension n_sample x n_features. Saved in data_path.

abstract_length_distribution_from2022.pdf: distribution of abstract length (Fig. 2.1 in the report).
Img (pdf format). Saved in data_path/img.

all_categories_from2022.pdf: Frequency of all subject categories (Fig. 2.2 in the report).
Img (pdf format). Saved in data_path/img.

top6_bottom10_categories_from2022.pdf: number of papers in the mostly and less frequent subject categories (Fig. 2.3 in the report),
Img (pdf format). Saved in data_path/img.

multiple_categories_from2022.pdf: number of papers with one or more categories associated (Fig. 2.4 in the report).
Img (pdf format). Saved in data_path/img.

---------- PCA_SciBert_emebedding.py ----------
This file extracts the last hidden layer of the pre-trained SciBert model and applies PCA method to obtain the final text representation.

INPUT
data_path, i.e. the path to the folder that contains the arXiv dataset "arxiv-metadata-oai-snapshot.json". 

OUTPUT
PCA_scibert_embedding_from2022.npy: PCA-SciBert (T) embedding
Numpy array of dimension n_samples x 325. Saved in data_path.

PCA_scibert_csl_embedding_from2022.npy: PCA-SciBert (CLS) embedding
Numpy array of dimension n_samples x 122. Saved in data_path.

PCA_SciBert_Embedding_mostfrequentclass_from2022.png': 2D projection of PCA-SciBert (T) embedding, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.5 in the report). 
Image (png format). Saved in data_path/img.

PCA_SciBert_cls_Embedding_mostfrequentclass_from2022.png': 2D projection of PCA-SciBert (CLS) embedding, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.7 in the report). 
Image (png format). Saved in data_path/img.

abstract_length_distribution_from2022.pd
---------- FineTuned_SciBert_embedding.py ----------
This files performs fine-tuning of the pre-trained SciBert model on the subject category classification task. Finally, it extracts the last hidden layer of the fine-tuned model as text representation.

INPUT
data_path, i.e. the path to the folder that contains the arXiv dataset "arxiv-metadata-oai-snapshot.json". 

OUTPUT
y_train.npy: subject category labels of the training set.
Numpy array of dimension n_train_samples x n_categories. Saved in data_path/finetuning.

y_test.npy: subject category labels of the test set.
Numpy array of dimension n_test_samples x n_categories. Saved in data_path/finetuning.

tuned_scibert_train_embedding_from2022.npy: Fine-Tuned SciBert (T) embedding of the training set.
Numpy array of dimension n_train_sapmles x 768. Saved in data_path/finetuning.

tuned_scibert_train_cls_embedding_from2022.npy: Fine-Tuned SciBert (CLS) embedding of the training set.
Numpy array of dimension n_train_sapmles x 768. Saved in data_path/finetuning.

tuned_scibert_test_embedding_from2022.npy: Fine-Tuned SciBert (T) embedding of the testing set.
Numpy array of dimension n_test_sapmles x 768. Saved in data_path/finetuning.

tuned_scibert_test_cls_embedding_from2022.npy: Fine-Tuned SciBert (CLS) embedding of the testing set.
Numpy array of dimension n_test_sapmles x 768. Saved in data_path/finetuning.

train_tuned_Embedding_mostfrequentclass_from2022.png: 2D projection of Fine-Tuned SciBert (T) embedding of the training set, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.7 in the report). 
Image (png format). Saved in data_path/img.

train_tuned_csl_Embedding_mostfrequentclass_from2022.png: 2D projection of Fine-Tuned SciBert (CLS) embedding of the training set, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.8 in the report). 
Image (png format). Saved in data_path/img.

test_tuned_Embedding_mostfrequentclass_from2022.png: 2D projection of Fine-Tuned SciBert (T) embedding of the testing set, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.9 in the report). 
Image (png format). Saved in data_path/img.

test_tuned_csl_Embedding_mostfrequentclass_from2022.png: 2D projection of Fine-Tuned SciBert (CLS) embedding of the testing set, where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.10 in the report). 
Image (png format). Saved in data_path/img.


---------- text_classification.py ----------
This file performs the unsupervised text classification based on K-Means algorithm by selecting the best number of classification categories based on Silhouette metric.
It also analyses the results on the testing set.

INPUT
data_path, i.e. the path to the folder that contains the arXiv dataset "arxiv-metadata-oai-snapshot.json". 

OUTPUT

tuned_scibert_cls_kmean_sil_from2022.png: silhouette score on the validation set at varying of the number of categories (Fig. 3.1 in the report).
Image (png format). Saved in data_path/img.

tuned_scibert_cls_kmeans_elbow_from2022.png: WCCS score on the validation set at varying of the number of categories (for elbow method).
Image (png format). Saved in data_path/img.

tuned_scibert_cls_kmeans_pred_k=32_from.npy: K-Means predicted labels on the testing set.
Numpy vector of dimension n_test_samples. Saved in data_path.

tuned_scibert_cls_kmeans_k=32_TSNE_from2022.png: K-Means clusters on testing set (Fig. 3.2 in the report).
Image (png format). Saved in data_path/img.

tuned_scibert_cls_kmean=32_count_categories.png: bar chart reporting the number of categories associated to cluster samples (Fig. 3.3 in the report).
Image (png format). Saved in data_path/img.

tuned_scibert_cls_test_classes_evaluation.txt: 3 most frequency categories (and their frequency) for each cluster.
Txt file. Saved in data_path.


## GENERAL INFORMATION

All the images are saved in data_path/img
Fine-tuning results are saved in data_path/finetuning
All other results are saved in data_path

