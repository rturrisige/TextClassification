"""
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
"""

import sys
import os
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1])
min_year = 2022

name = 'tuned_scibert_cls'  # change it to "tuned_scibert" to perform text classification on Fine-Tuned SciBert (T)
learn_embedding = np.load(data_path + 'finetuning/' + name + '_train_embedding_from' + str(min_year) + '.npy')
test_embedding = np.load(data_path + 'finetuning/' + name + '_test_embedding_from' + str(min_year) + '.npy')
y_learn = np.load(data_path + 'finetuning/' + 'y_train.npy')
y_test = np.load(data_path + 'finetuning/' + 'y_test.npy')

train_embedding, val_embedding,\
y_train, y_val = train_test_split(learn_embedding,
                                   y_learn,
                                   test_size=0.2,
                                   random_state=42,
                                   stratify=y_learn)

##
# Looking for the best number of categories

k_values = np.arange(2, 51)

# Silhouette method
sil_values = find_best_k(train_embedding, val_embedding, k_range=k_values)
kbest, ymax = k_values[np.argmax(sil_values)], np.max(sil_values)

plt.figure(figsize=(16, 9))
plt.plot(sil_values, "o-")
plt.plot(np.argmax(sil_values), ymax, '*', color='red')
plt.xlabel("N clusters", fontsize=25)
plt.ylabel("silhouette", fontsize=25)
plt.xticks(np.arange(0, len(k_values), 1), k_values)
plt.yticks(fontsize=25)
plt.gca().xaxis.grid(True)
plt.savefig(data_path + 'img/' + name + '_kmean_sil_from' + str(min_year) + '.png', bbox_inches='tight')

print('Silhoutte method: Best k=', kbest)

# Elbow method
cluster_errors = compute_elbow_kmeans(train_embedding, val_embedding, k_range=k_values)
elbow = KneeLocator(k_values, cluster_errors, curve='convex', direction='decreasing')
print('Elbow at:', elbow.knee)

plt.figure(figsize=(16, 9))
plt.plot(cluster_errors, "o-")
plt.xlabel("k clusters", fontsize=25)
plt.ylabel("Sum Sqr distances from mean", fontsize=25)
plt.yticks(fontsize=15)
plt.xticks(np.arange(0, len(k_values), 1), k_values)
plt.savefig(data_path + 'img/' + name + '_kmeans_elbow_from' + str(min_year) + '.png', bbox_inches='tight')


##
# Clustering with best number of clusters

# Train K-Means with kbest classes
kmeans = KMeans(n_clusters=kbest, random_state=42).fit(learn_embedding)

# Evaluation on testing set
test_pred = kmeans.predict(test_embedding)
print('Silhouette on testing:', metrics.silhouette_score(test_embedding, test_pred))
np.save(data_path + name + '_kmeans_pred_k=' + str(kbest) + '_from' + str(min_year) + '.npy', test_pred)

# Cluster visualization in 2D
cluster_visualization(test_embedding, test_pred, data_path, name, kbest)


##
# Check most frequent categories within each cluster

# plot bar chart for category frequency in each cluster
plot_categories_in_clusters(test_pred, y_test, data_path, name, kbest, nrow=8, ncol=4)

unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy',
                                 allow_pickle=True).item()

macro_cat = []
logfile = open(data_path + name + '_test_classes_evaluation.txt', 'w')
for c in range(kbest):
    papers_c = np.where(test_pred == c)[0]
    labels_c = np.array(y_test)[papers_c]
    count_c = np.sum(labels_c, 0)
    sorted_indices = np.argsort(count_c)[::-1]
    logfile.write('\nCluster ' + str(c) + '\n')
    macro = []
    for i in sorted_indices[:3]:
        cat_name = list(unique_categories_dict.keys())[i]
        logfile.write('{} ({}):{:d}. '.format(category_map[cat_name], cat_name, int(count_c[i])))
        if count_c[i] > 9:
            macro.append(cat_name.split('.')[0].split('-')[0])
    macro_cat.append(macro)
    logfile.write('\n')
    logfile.flush()


one_cat, two_cat, three_cat= 0, 0, 0
for m in macro_cat:
    unique_macro_c = len(list(set(m)))
    if unique_macro_c == 1:
        one_cat += 1
    elif unique_macro_c == 2:
        two_cat += 1
    else:
        three_cat += 1

logfile.write('\nOne category or completely matching categories: {}%\n'.format(one_cat/(one_cat+two_cat+three_cat)))
logfile.write('Two main categories: {}%\n'.format(two_cat/(one_cat+two_cat+three_cat)))
logfile.write('Three distinct categories: {}%'.format(three_cat/(one_cat+two_cat+three_cat)))
logfile.close()


