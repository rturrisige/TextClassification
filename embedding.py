import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import sys
import os
from sklearn.manifold import TSNE
import ast
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1]) # '/home/rosannaturrisi/storage/NLP/'
min_year = 2022
data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
categories_labels = np.load(data_path + 'categories_labels.npy')


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

# most and less frequent category plot on the 2D-projection of the embedding

def scatter_one_class(embedding_2D, idx, label):
    all_others = [i for i in range(mf_cat_papers.shape[0]) if i not in idx]
    plt.scatter(embedding_2D[all_others, 0], embedding_2D[all_others, 1], color='darkgrey')
    plt.scatter(embedding_2D[idx, 0], embedding_2D[idx, 1], color='lightseagreen', label=label)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=14)


category_count = np.sum(categories_labels, 0)

mf_cat = np.argsort(category_count)[::-1][:6]
mf_cat_papers = categories_labels[:, mf_cat]
indices = []
for n in range(6):
    indices.append(list(np.where(mf_cat_papers[:, n] == 1)[0]))

label_names = ['1st freq.', '2nd freq.', '3nd freq.', '4th freq.', '5th freq.', '6th freq.']

plt.figure(figsize=(20, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(embedding_2D, indices[i], label_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=20)
plt.savefig(data_path + 'Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

