"""
This file contains useful functions related to categories visualisation, embedding creation and visualisation,
clustering training and evaluation. It also includes a dictionary that maps the arXiv category labels (e.g., 'astro-ph')
into the corresponding subject name (e.g., 'Astrophysics').
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.manifold import TSNE

# Abstract length

def plot_wdist(data_path, abstract_lengths, min_year=2022, format='.png'):
    plt.figure(figsize=(10, 5))
    plt.hist(np.sort(abstract_lengths), bins=20)
    plt.xlabel('Number of words', fontsize=20)
    plt.ylabel('Number of papers', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig(data_path + 'img/abstract_length_distribution_from' + str(min_year) + format, bbox_inches='tight')

# Categories visualization

def plot_all_categories(unique_categories, category_count, saver_path, min_year, format='.png'):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(16, 9))
    plt.title('Categories Frequency', fontsize=30)
    plt.ylabel('Number of Papers', fontsize=20)
    plt.xlabel('Subject Category', fontsize=20)
    plt.bar([unique_categories[i] for i in indices], sorted(category_count)[::-1])
    plt.xticks([i for i in range(0, len(unique_categories), 10)], [i for i in range(0, len(unique_categories), 10)])
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(saver_path + 'img/all_categories_from' + str(min_year) + format, bbox_inches='tight')


def plot_top_categories(unique_categories, category_count, saver_path, min_year, ntop=20, format='.png'):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(16, 9))
    plt.title('Top ' + str(ntop) + ' Categories', fontsize=30)
    plt.ylabel('Number of Papers', fontsize=25)
    plt.xlabel('Labeled Paper Category', fontsize=25)
    plt.bar([unique_categories[i] for i in indices[:ntop]], sorted(category_count)[::-1][:ntop])
    plt.xticks(ha="right", rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(saver_path + 'top' + str(ntop) +'_categories_from' + str(min_year) + format, bbox_inches='tight')


def plot_bottom_categories(unique_categories, category_count, saver_path, min_year, nbottom=30, format='.png'):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(16, 9))
    plt.title('Bottom ' + str(nbottom) + ' Categories', fontsize=30)
    plt.ylabel('Number of Papers', fontsize=25)
    plt.xlabel('Subject Category', fontsize=25)
    bottom_frequencies = sorted(category_count)[::-1][-nbottom:]
    plt.xticks(np.arange(0, len(bottom_frequencies)))
    plt.xticks(ha="right", rotation=45, fontsize=20)
    plt.bar([unique_categories[i] for i in indices[-nbottom:]], bottom_frequencies)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(saver_path + 'img/bottom' + str(nbottom) + '_categories_from' + str(min_year) + format, bbox_inches='tight')


def plot_top_bottom_categories(unique_categories, category_count, saver_path, min_year, ntop=6, nbottom=10, format='.png'):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.title('Top ' + str(ntop) + ' Categories', fontsize=30)
    plt.ylabel('Number of Papers', fontsize=25)
    plt.xlabel('Subject Category', fontsize=25)
    plt.bar([unique_categories[i] for i in indices[:ntop]], sorted(category_count)[::-1][:ntop])
    plt.xticks(ha="right", rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(122)
    plt.title('Bottom ' + str(nbottom) + ' Categories', fontsize=30)
    plt.xlabel('Subject Category', fontsize=25)
    bottom_frequencies = sorted(category_count)[::-1][-nbottom:]
    plt.xticks(np.arange(0, len(bottom_frequencies)))
    plt.xticks(ha="right", rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.bar([unique_categories[i] for i in indices[-nbottom:]], bottom_frequencies)
    plt.subplots_adjust(bottom=0.3, hspace=0.1)
    plt.savefig(saver_path + 'img/top' + str(ntop) + '_bottom' + str(nbottom) + '_categories_from' + str(min_year) + format,
                bbox_inches='tight')


def plot_multiple_categories(nlabel, counts, saver_path, min_year):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    fig.suptitle('Multi-Categories Counts', fontsize=20)
    plt.ylabel('Number of Papers', fontsize=14)
    plt.xlabel('Number of Categories', fontsize=14)
    plt.setp(ax.get_xticklabels(), ha="right", fontsize=14)  # Specify a rotation for the tick labels
    ax.set_xticks(nlabel)
    plt.bar(nlabel, counts)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(saver_path + 'img/multiple_categories_from' + str(min_year) + '.png', bbox_inches='tight')


##
# SciBert Tokenizer

def scibert_encode(data, tokenizer, max_length=100):
    input_ids, attention_masks = [], []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


##
# Embedding visualization

def scatter_one_class(embedding_2D, idx, label):
    all_others = [i for i in range(embedding_2D.shape[0]) if i not in idx]
    plt.scatter(embedding_2D[all_others, 0], embedding_2D[all_others, 1], color='darkgrey')
    plt.scatter(embedding_2D[idx, 0], embedding_2D[idx, 1], color='lightseagreen', label=label)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=20)


def n_most_frequent_categories(categories_labels, unique_category, n):
    category_count = np.sum(categories_labels, 0)
    mf_cat = np.argsort(category_count)[::-1][:n]
    mf_cat_papers = categories_labels[:, mf_cat]
    names = np.array(unique_category)[mf_cat]
    indices = []
    for n in range(6):
        indices.append(list(np.where(mf_cat_papers[:, n] == 1)[0]))
    return indices, names

##
# Clustering

def find_best_k(X_train, X_val, k_range, random_state=42):
    """Find the best k via grid search optmizing the Silhouette score."""
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_train)
        val_pred = kmeans.predict(X_val)
        #sil = metrics.silhouette_score(X, kmeans.labels_.ravel())
        sil = metrics.silhouette_score(X_val, val_pred)
        print('k=' + str(k) + ', silhouette score=' + str(sil))
        silhouette_scores.append(sil)
    print('')
    return silhouette_scores

def compute_elbow_kmeans(X_train, X_val, k_range):
    r_seed = 24
    cluster_errors = []
    for n_clusters in k_range:
        pipe_pca_kmean = Pipeline([("cluster", KMeans(n_clusters=n_clusters, random_state=r_seed, verbose=0))])
        pipe_pca_kmean.fit(X_train)
        pipe_pca_kmean.predict(X_val)
        cluster_errors.append(pipe_pca_kmean.named_steps["cluster"].inertia_)
    return cluster_errors


# cluster visualization

def cluster_visualization(test_embedding, test_pred, data_path, name, kbest, min_year=2022):
    tsne = TSNE(random_state=42, init='pca')
    embedding_2D = tsne.fit_transform(test_embedding)

    # plot
    plt.figure(figsize=(16, 9))
    sns.set(rc={'figure.figsize': (15, 15)})
    palette = sns.hls_palette(kbest, l=.4, s=.9)  # colors
    sns.scatterplot(x=embedding_2D[:,0], y=embedding_2D[:,1], hue=test_pred, legend='full',
                    palette=palette)

    plt.title('K-Means classes', fontsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(data_path + 'img/' + name + '_kmeans_k=' + str(kbest) + '_TSNE_from' + str(min_year) + '.png',
                bbox_inches='tight')


def plot_categories_in_clusters(test_pred, y_test, data_path, name, kbest, nrow=8, ncol=4):
    plt.figure(figsize=(18, 20))
    i = 1
    for c in range(max(test_pred) + 1):
        papers_c = np.where(test_pred == c)[0]
        labels_c = np.array(y_test)[papers_c]
        count_c = np.sum(labels_c, 0)
        plt.subplot(nrow, ncol, i)
        plt.title('Cluster ' + str(i-1), fontsize=20)
        plt.bar(np.arange(1, 41), list(count_c))
        if (i-1) % ncol == 0:
            plt.yticks(fontsize=20)
            plt.ylabel('N. papers', fontsize=20, labelpad=10)
        if i > ((nrow-1)*ncol) :
            plt.xticks(np.arange(0, 41, 10), fontsize=20)
            plt.xlabel('Subject Category', fontsize=20, labelpad=10)
        else:
            plt.xticks(np.arange(0, 41, 10), [])
        i += 1

    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.savefig(data_path + 'img/' + name + '_kmeans_k='+str(kbest) + '_count_categories.png', bbox_inches='tight')

##
# Label-to-category map


category_map = {'astro-ph': 'Astrophysics',
                'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
                'astro-ph.EP': 'Earth and Planetary Astrophysics',
                'astro-ph.GA': 'Astrophysics of Galaxies',
                'astro-ph.HE': 'High Energy Astrophysical Phenomena',
                'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
                'astro-ph.SR': 'Solar and Stellar Astrophysics',
                'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
                'cond-mat': 'Condensed Matter',
                'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
                'cond-mat.mtrl-sci': 'Materials Science',
                'cond-mat.other': 'Other Condensed Matter',
                'cond-mat.quant-gas': 'Quantum Gases',
                'cond-mat.soft': 'Soft Condensed Matter',
                'cond-mat.stat-mech': 'Statistical Mechanics',
                'cond-mat.str-el': 'Strongly Correlated Electrons',
                'cond-mat.supr-con': 'Superconductivity',
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control',
                'econ.EM': 'Econometrics',
                'eess.AS': 'Audio and Speech Processing',
                'eess.IV': 'Image and Video Processing',
                'eess.SP': 'Signal Processing',
                'gr-qc': 'General Relativity and Quantum Cosmology',
                'hep-ex': 'High Energy Physics - Experiment',
                'hep-lat': 'High Energy Physics - Lattice',
                'hep-ph': 'High Energy Physics - Phenomenology',
                'hep-th': 'High Energy Physics - Theory',
                'math.AC': 'Commutative Algebra',
                'math.AG': 'Algebraic Geometry',
                'math.AP': 'Analysis of PDEs',
                'math.AT': 'Algebraic Topology',
                'math.CA': 'Classical Analysis and ODEs',
                'math.CO': 'Combinatorics',
                'math.CT': 'Category Theory',
                'math.CV': 'Complex Variables',
                'math.DG': 'Differential Geometry',
                'math.DS': 'Dynamical Systems',
                'math.FA': 'Functional Analysis',
                'math.GM': 'General Mathematics',
                'math.GN': 'General Topology',
                'math.GR': 'Group Theory',
                'math.GT': 'Geometric Topology',
                'math.HO': 'History and Overview',
                'math.IT': 'Information Theory',
                'math.KT': 'K-Theory and Homology',
                'math.LO': 'Logic',
                'math.MG': 'Metric Geometry',
                'math.MP': 'Mathematical Physics',
                'math.NA': 'Numerical Analysis',
                'math.NT': 'Number Theory',
                'math.OA': 'Operator Algebras',
                'math.OC': 'Optimization and Control',
                'math.PR': 'Probability',
                'math.QA': 'Quantum Algebra',
                'math.RA': 'Rings and Algebras',
                'math.RT': 'Representation Theory',
                'math.SG': 'Symplectic Geometry',
                'math.SP': 'Spectral Theory',
                'math.ST': 'Statistics Theory',
                'math-ph': 'Mathematical Physics',
                'nlin.AO': 'Adaptation and Self-Organizing Systems',
                'nlin.CD': 'Chaotic Dynamics',
                'nlin.CG': 'Cellular Automata and Lattice Gases',
                'nlin.PS': 'Pattern Formation and Solitons',
                'nlin.SI': 'Exactly Solvable and Integrable Systems',
                'nucl-ex': 'Nuclear Experiment',
                'nucl-th': 'Nuclear Theory',
                'physics.acc-ph': 'Accelerator Physics',
                'physics.ao-ph': 'Atmospheric and Oceanic Physics',
                'physics.app-ph': 'Applied Physics',
                'physics.atm-clus': 'Atomic and Molecular Clusters',
                'physics.atom-ph': 'Atomic Physics',
                'physics.bio-ph': 'Biological Physics',
                'physics.chem-ph': 'Chemical Physics',
                'physics.class-ph': 'Classical Physics',
                'physics.comp-ph': 'Computational Physics',
                'physics.data-an': 'Data Analysis, Statistics and Probability',
                'physics.ed-ph': 'Physics Education',
                'physics.flu-dyn': 'Fluid Dynamics',
                'physics.gen-ph': 'General Physics',
                'physics.geo-ph': 'Geophysics',
                'physics.hist-ph': 'History and Philosophy of Physics',
                'physics.ins-det': 'Instrumentation and Detectors',
                'physics.med-ph': 'Medical Physics',
                'physics.optics': 'Optics',
                'physics.plasm-ph': 'Plasma Physics',
                'physics.pop-ph': 'Popular Physics',
                'physics.soc-ph': 'Physics and Society',
                'physics.space-ph': 'Space Physics',
                'q-bio.BM': 'Biomolecules',
                'q-bio.CB': 'Cell Behavior',
                'q-bio.GN': 'Genomics',
                'q-bio.MN': 'Molecular Networks',
                'q-bio.NC': 'Neurons and Cognition',
                'q-bio.OT': 'Other Quantitative Biology',
                'q-bio.PE': 'Populations and Evolution',
                'q-bio.QM': 'Quantitative Methods',
                'q-bio.SC': 'Subcellular Processes',
                'q-bio.TO': 'Tissues and Organs',
                'q-fin.CP': 'Computational Finance',
                'q-fin.EC': 'Economics',
                'q-fin.GN': 'General Finance',
                'q-fin.MF': 'Mathematical Finance',
                'q-fin.PM': 'Portfolio Management',
                'q-fin.PR': 'Pricing of Securities',
                'q-fin.RM': 'Risk Management',
                'q-fin.ST': 'Statistical Finance',
                'q-fin.TR': 'Trading and Market Microstructure',
                'quant-ph': 'Quantum Physics',
                'stat.AP': 'Applications',
                'stat.CO': 'Computation',
                'stat.ME': 'Methodology',
                'stat.ML': 'Machine Learning',
                'stat.OT': 'Other Statistics',
                'stat.TH': 'Statistics Theory'}