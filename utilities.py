from matplotlib import pyplot as plt
import numpy as np

# Categories visualization


def plot_all_categories(unique_categories, category_count, saver_path, min_year):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(16, 9))
    plt.title('Categories Frequency', fontsize=20)
    plt.ylabel('Number of Papers', fontsize=14)
    plt.xlabel('Paper Category', fontsize=14)
    plt.bar([unique_categories[i] for i in indices], sorted(category_count)[::-1])
    plt.xticks([i for i in range(0, len(unique_categories), 10)], [i for i in range(0, len(unique_categories), 10)])
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(saver_path + 'all_categories_from' + str(min_year) + '.png')


def plot_top_categories(unique_categories, category_count, saver_path, min_year, ntop=20):
    indices = np.argsort(category_count)[::-1]
    plt.figure(figsize=(16, 9))
    plt.title('Top ' + str(ntop) + ' Categories', fontsize=20)
    plt.ylabel('Number of Papers', fontsize=14)
    plt.xlabel('Labeled Paper Category', fontsize=14)
    plt.bar([unique_categories[i] for i in indices[:ntop]], sorted(category_count)[::-1][:ntop])
    plt.xticks(ha="right", rotation=45, fontsize=14)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(saver_path + 'top' + str(ntop) +'_categories_from' + str(min_year) + '.png')


def plot_bottom_categories(unique_categories, category_count, saver_path, min_year, nbottom=30):
    indices = np.argsort(category_count)[::-1]
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Bottom ' + str(nbottom) + ' Categories', fontsize=20)
    plt.ylabel('Number of Papers', fontsize=14)
    plt.xlabel('Labeled Paper Category', fontsize=14)
    bottom_frequencies = sorted(category_count)[::-1][-nbottom:]
    plt.xticks(np.arange(0, max(bottom_frequencies) + 0.1))
    plt.xticks(ha="right", rotation=45, fontsize=14)
    plt.bar([unique_categories[i] for i in indices[-nbottom:]], bottom_frequencies)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(saver_path + 'bottom' + str(nbottom) + '_categories_from' + str(min_year) + '.png')


def plot_multiple_categories(nlabel, counts, saver_path, min_year):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    fig.suptitle('Multi-Categories Counts', fontsize=20)
    plt.ylabel('Number of Papers', fontsize=14)
    plt.xlabel('Number of Categories', fontsize=14)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14)  # Specify a rotation for the tick labels
    ax.set_xticks(nlabel)
    plt.bar(nlabel, counts)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(saver_path + 'multiple_categories_from' + str(min_year) + '.png')


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


import tensorflow as tf


def create_model(embedding_model, config):
    input_ids = tf.keras.Input(shape=(config.input_size,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(config.input_size,), dtype='int32')
    output = embedding_model([input_ids, attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(config.n_dense, activation='relu')(output)
    output = tf.keras.layers.Dropout(config.dropout)(output)
    output = tf.keras.layers.Dense(config.n_classes, activation='sigmoid')(output)
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    return model


class Classification_model(tf.keras.Model):
  def __init__(self, config, embedding_model):
    super().__init__()
    self.embedding_model = embedding_model
    self.dense1 = tf.keras.layers.Dense(config.n_dense, activation='relu')
    self.dropout = tf.keras.layers.Dropout(config.dropout)
    self.dense2 = tf.keras.layers.Dense(config.n_classes, activation='sigmoid')

  def call(self, input_ids, attention_masks, return_embeddings=False):
    embeddings = self.embedding_model([input_ids, attention_masks])
    output = embeddings[1]
    output = self.dense1(output)
    output = self.dropout(output)
    output = self.dense2(output)
    if return_embeddings:
        return embeddings, output
    else:
        return output