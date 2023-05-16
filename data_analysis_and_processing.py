"""
This file selects a subset of the arXiv dataset (papers published after 2022),
performs category labels and abstract lenght analysis, and abstract processing.

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
"""

import json
import pandas as pd
# NLP processing packages:
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg
import string
from tqdm import tqdm
import sys
import os
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1])

if not os.path.exists(data_path + 'img/'):
    os.makedirs(data_path + 'img/')

metadata = []
with open(data_path + "arxiv-metadata-oai-snapshot.json", 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

print('Data loaded from json file.')

data = pd.DataFrame(metadata)
data.info()

# select a subset for computational reason
min_year = 2022
data = data[data['journal-ref'].str.contains('None') == False]
data['journal-ref'] = data['journal-ref'].str.replace(r'[()].:,-', " ", regex=False)
for c in '.:,-()':
    data['journal-ref'] = data['journal-ref'].str.replace(c, " ", regex=False)

recent_papers = []
i = 0
for ref in data['journal-ref']:
    y = ref.strip().split(' ')[-1]
    if y.isdigit() and int(y) > min_year:
        recent_papers.append(i)
    i += 1

data = data.iloc[recent_papers]
print('Subset selection. Papers from:', min_year)

# select useful metadata
keep_col = ['id', 'authors', 'title', 'categories', 'abstract']
data = data[keep_col]
print('Kept information:', keep_col)

# remove duplicates and withdrawn papers
data = data[data['abstract'].str.contains('paper has been withdrawn') == False]
data.drop_duplicates(['abstract',], inplace=True)
print('Duplicates and withdrawn papers are removed.\n')

# remove categories with few papers
shortlisted_categories = data['categories'].value_counts().reset_index(name="count").query("count > 250")["index"].tolist()
data = data[data["categories"].isin(shortlisted_categories)].reset_index(drop=True)
print('Categories with less thant 250 papers are removed.\n')

print('Final corpus:')
data.head()

print('\nN. of texts:', data.shape[0])
print('')

##
# #Abstract analysis

# remove space symbols from text
data['abstract'] = data['abstract'].str.replace('\n', ' ')
data['abstract'] = data['abstract'].str.replace('\s', ' ', regex=True)

abstract_lengths = [len(t.split()) for t in data['abstract']]
tokeep = [i for i in range(len(abstract_lengths)) if abstract_lengths[i] > 30]

data = data.iloc[tokeep]
abstract_lengths = np.array(abstract_lengths)[tokeep]
print('Too short abstracts are removed.')


abstract_statistics = pd.DataFrame(abstract_lengths).describe()
print(abstract_statistics)
abstract_statistics.to_csv(data_path + 'abstract_length_statistics_data_from' + str(min_year) + '.csv')

plot_wdist(data_path, abstract_lengths, format='.pdf')

#################
# CREATE LABELS #
# ###############

categories_list = [x.split(' ') for x in data['categories']]
unique_categories = list(set([i for l in categories_list for i in l]))
print('N. of categories:', len(unique_categories))
print('')

unique_categories_dict = {}
for i, c in enumerate(unique_categories):
    unique_categories_dict[c] = i

np.save(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy', unique_categories_dict)

categories_labels = []
for paper_categories in categories_list:
    label = np.zeros([len(unique_categories)])
    for c in paper_categories:
        label[unique_categories_dict[c]] = 1
    categories_labels.append(list(label))

data['labels'] = categories_labels
np.save(data_path + 'categories_labels.npy', categories_labels)

# LABEL VISUALIZATION

category_count = np.sum(categories_labels, 0)
plot_all_categories(unique_categories, category_count, data_path, min_year, format='.pdf')
plot_top_bottom_categories(unique_categories, category_count, data_path, min_year, format='.pdf')
multilabel_count = np.sum(categories_labels, 1)
nlabel, counts = np.unique(multilabel_count, return_counts=True)
plot_multiple_categories(nlabel, counts, data_path, min_year)


####################
# TEXT PROCESSING  #
# ##################

print('Abstract processing...')

# Lemmatization
parser = en_core_sci_lg.load()  # specific package for scientific papers
parser.max_length = 7000000  # Limit the size of the parser
punctuations = string.punctuation  # list of punctuation to remove from text
stopwords = list(STOP_WORDS)  # list of stop words (e.g. 'now', 'top') to remove from text


def data_processing(sentence):
    text = parser(sentence)
    # transform to lowercase and then split the sentence
    text = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text]
    # remove stop words and punctuation
    text = [word for word in text if word not in stopwords and word not in punctuations]
    text = " ".join([i for i in text])
    return text


tqdm.pandas()
data['processed_abstract'] = data["abstract"].progress_apply(data_processing)
data.to_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
print('Data frame with preprocessed abstract saved at', data_path)