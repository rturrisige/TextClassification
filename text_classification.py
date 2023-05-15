import pandas as pd
import sys, os
sys.path.append(os.getcwd() + '/')
from utilities import *
from sklearn.manifold import TSNE
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.model_selection import train_test_split
from kneed import KneeLocator

################
# DATA READING #
# ##############

data_path = '/home/rosannaturrisi/storage/NLP/'
min_year = 2022
#data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')

name = 'tuned_scibert'
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
plt.xlabel("k clusters", fontsize=25)
plt.ylabel("silhouette", fontsize=25)
plt.xticks(np.arange(0, len(k_values), 1), k_values)
plt.yticks(fontsize=15)
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
# Clustering with best number of k

kmeans = KMeans(n_clusters=kbest, random_state=42).fit(learn_embedding)
test_pred = kmeans.predict(test_embedding)
np.save(data_path + name + '_kmeans_pred_k='+ str(kbest) + '_from' + str(min_year) + '.npy', test_pred)
print('Silhouette on testing:', metrics.silhouette_score(test_embedding, test_pred))
##
# Cluster visualization in 2D

tsne = TSNE(random_state=42, init='pca')
embedding_2D = tsne.fit_transform(test_embedding)

# plot
plt.figure(figsize=(16, 9))
sns.set(rc={'figure.figsize': (15, 15)})
palette = sns.hls_palette(kbest, l=.4, s=.9)  # colors
sns.scatterplot(x=embedding_2D[:,0], y=embedding_2D[:,1], hue=test_pred, legend='full',
                palette=palette)

plt.title('TSNE with Kmeans Labels', fontsize=30)
plt.xticks([])
plt.yticks([])
plt.savefig(data_path + 'img/' + name + '_kmeans_k=' + str(kbest) + '_TSNE_from' + str(min_year) + '.png',
            bbox_inches='tight')

##

# Cluster evaluation

plt.figure(figsize=(18, 20))
i = 1
for c in range(max(test_pred) + 1):
    papers_c = np.where(test_pred == c)[0]
    labels_c = np.array(y_test)[papers_c]
    count_c = np.sum(labels_c, 0)
    plt.subplot(7, 5, i)
    plt.title('Cluster ' + str(i-1), fontsize=20)
    plt.bar(np.arange(1, 41), list(count_c))
    plt.ylim(0, 601)
    if (i-1) % 5 == 0:
        plt.yticks(np.arange(0, 601, 100), fontsize=20)
        plt.ylabel('N. papers', fontsize=20, labelpad=10)
    else:
        plt.yticks(np.arange(0, 601, 100), [])
    if i > 30:
        plt.xticks(np.arange(0, 41, 10), fontsize=20)
        plt.xlabel('Subject Category', fontsize=20, labelpad=10)
    else:
        plt.xticks(np.arange(0, 41, 10), [])
    i += 1

plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.savefig(data_path + 'img/' + name + '_kmeans_k='+str(kbest) + '_count_categories.png', bbox_inches='tight')

##
# Check most frequent categories within each cluster
unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy',
                                 allow_pickle=True).item()

logfile = open(data_path + name + 'test_classes_evaluation.txt', 'w')
for c in range(kbest):
    papers_c = np.where(test_pred == c)[0]
    labels_c = np.array(y_test)[papers_c]
    count_c = np.sum(labels_c, 0)
    sorted_indices = np.argsort(count_c)[::-1]
    logfile.write('\nCluster ' + str(c) + '\n')
    for i in sorted_indices[:3]:
        name = list(unique_categories_dict.keys())[i]
        logfile.write('{} ({}):{:d}. '.format(category_map[name], name, int(count_c[i])))
    logfile.write('\n')
    logfile.flush()

logfile.close()