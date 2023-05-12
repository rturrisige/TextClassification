import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import sys, os
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1]) # '/home/rosannaturrisi/storage/NLP/'
min_year = 2022
data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.py', )


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
for id, atm in input_dataset:
    output = model.predict([id, atm])[0][:, 0, :]
    bert_embedding = np.concatenate([bert_embedding, output], 0) if bert_embedding.any() else output

np.save(data_path + 'bert_embedding_from' + str(min_year) + '.npy', bert_embedding)
##

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of the variance
embedding = pca.fit_transform(bert_embedding)
print('Embedding shape:', embedding.shape)

np.save(data_path + 'PCA_bert_embedding_from' + str(min_year) + '.npy', embedding)
