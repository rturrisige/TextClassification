from matplotlib import pyplot as plt
import numpy as np

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
    plt.savefig(saver_path + 'all_categories_from' + str(min_year) + format, bbox_inches='tight')


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
    plt.savefig(saver_path + 'bottom' + str(nbottom) + '_categories_from' + str(min_year) + format, bbox_inches='tight')


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
    plt.savefig(saver_path + 'top' + str(ntop) + '_bottom' + str(nbottom) + '_categories_from' + str(min_year) + format,
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
    plt.savefig(saver_path + 'multiple_categories_from' + str(min_year) + '.png', bbox_inches='tight')


# Encoding


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