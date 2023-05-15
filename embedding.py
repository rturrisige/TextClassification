import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import sys
import os
from sklearn.manifold import TSNE
import ast
sys.path.append(os.getcwd() + '/')
from utilities import *
import tensorflow as tf

################
# DATA READING #
# ##############

data_path = str(sys.argv[1]) # '/home/rosannaturrisi/storage/NLP/'
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
for id, atm in input_dataset:
    output = model.predict([id, atm])[1] #[0][:, 0, :]
    bert_embedding = np.concatenate([bert_embedding, output], 0) if bert_embedding.any() else output

np.save(data_path + 'scibert_embedding_from' + str(min_year) + '.npy', bert_embedding)
##

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of the variance
embedding = pca.fit_transform(bert_embedding)
print('Embedding shape:', embedding.shape)

np.save(data_path + 'PCA_scibert_embedding_from' + str(min_year) + '.npy', embedding)


##
# Qualitative evaluation

tsne = TSNE(random_state=42, init='pca')
embedding_2D = tsne.fit_transform(embedding)

# most frequent categories plot on the 2D-projection of the embedding
indices, cat_names = n_most_frequent_categories(categories_labels,  list(unique_categories_dict.keys()), 6)

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(embedding, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/SciBert_cls_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

