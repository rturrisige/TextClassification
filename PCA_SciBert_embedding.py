"""
This file extracts the last hidden layer of the pre-trained SciBert model and applies PCA method to obtain the final text representation.

INPUT
data_path, i.e. the path to the folder that contains the arXiv dataset "arxiv-metadata-oai-snapshot.json".

OUTPUT
PCA_scibert_embedding_from2022.npy: PCA-SciBert (T) embedding
Numpy array of dimension n_samples x 325. Saved in data_path.

PCA_scibert_csl_embedding_from2022.npy: PCA-SciBert (CLS) embedding
Numpy array of dimension n_samples x 122. Saved in data_path.

PCA_SciBert_Embedding_mostfrequentclass_from2022.png': 2D projection of PCA-SciBert (T) embedding,
where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.5 of the report).
Image (png format). Saved in data_path/img.

PCA_SciBert_cls_Embedding_mostfrequentclass_from2022.png': 2D projection of PCA-SciBert (CLS) embedding,
where samples belonging to the 6 most frequent categories are labelled in green (Fig. 2.5 of the report).
Image (png format). Saved in data_path/img.
"""

import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import sys
import os
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1])
min_year = 2022
data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
categories_labels = np.load(data_path + 'categories_labels.npy')
unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy',
                                 allow_pickle=True).item()

####################
# TEXT EMBEDDING   #
# ##################

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = TFAutoModel.from_pretrained('allenai/scibert_scivocab_uncased', from_pt=True)

X = list(data['processed_abstract'])
input_ids, attention_masks = scibert_encode(X, tokenizer, max_length=157)
del X

input_dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))
input_dataset = input_dataset.batch(batch_size=500, drop_remainder=False)

bert_embedding = np.array([])
bert_cls_embedding = np.array([])
for id, atm in input_dataset:
    output = model.predict([id, atm])
    bert_embedding = np.concatenate([bert_embedding, output[0][:, 0, :]], 0) if bert_embedding.any() else output[0][:, 0, :]
    bert_cls_embedding = np.concatenate([bert_embedding, output[1]], 0) if bert_embedding.any() else output[1]


pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of the variance
embedding = pca.fit_transform(bert_embedding)
cls_embedding = pca.fit_transform(bert_cls_embedding)

print('Embedding shape:', embedding.shape)
print('Embedding shape:', cls_embedding.shape)

np.save(data_path + 'PCA_scibert_embedding_from' + str(min_year) + '.npy', embedding)
np.save(data_path + 'PCA_scibert_cls_embedding_from' + str(min_year) + '.npy', cls_embedding)

##
# Qualitative evaluation

# most frequent categories plot on the 2D-projection of the embedding
indices, cat_names = n_most_frequent_categories(categories_labels,  list(unique_categories_dict.keys()), 6)

tsne = TSNE(random_state=42, init='pca')

# PCA-SciBert (T)
embedding_2D = tsne.fit_transform(embedding)

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(embedding, indices[i], cat_names[i])

plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/PCA_SciBert_Embedding_mostfrequentclass_from' + str(min_year) + '.png',
            bbox_inches='tight')

# PCA-SciBert (CLS)
cls_embedding_2D = tsne.fit_transform(cls_embedding)

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(cls_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/PCA_SciBert_cls_Embedding_mostfrequentclass_from' + str(min_year) + '.png',
            bbox_inches='tight')