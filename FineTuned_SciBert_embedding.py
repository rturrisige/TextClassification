import pandas as pd
import sys
import os
from sklearn.manifold import TSNE
import ast
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
sys.path.append(os.getcwd() + '/')
from utilities import *

################
# DATA READING #
# ##############

data_path = str(sys.argv[1])  # '/home/rosannaturrisi/storage/NLP/'

if not os.path.exists(data_path + 'finetuning/'):
    os.makedirs(data_path + 'finetuning/')

min_year = 2022
data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy',
                                 allow_pickle=True).item()

categories_labels = np.load(data_path + 'categories_labels.npy')

##

class Configuration(object):
    def __init__(self):
        # self.nexp = int(sys.argv[1])
        # Network parameters
        self.input_size = 157
        self.n_dense = 32
        self.n_classes = 40
        # Training parameters:
        self.batch_size = 32
        self.num_epochs = 8  # maximum number of iterations
        self.lr = 2e-5  # learning rate
        self.patience = 1  # patience for early stopping
        self.criterion = 'categorical_crossentropy'  # loss
        self.n_validation = 0.2  # validation set proportion
        self.n_test = 0.1  # testing set proportion
        self.dropout = 0.1


config = Configuration()

####################
# TEXT EMBEDDING   #
# ##################

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = TFAutoModel.from_pretrained('allenai/scibert_scivocab_uncased', from_pt=True)

X = list(data['processed_abstract'])

input_ids, attention_masks = scibert_encode(X, tokenizer, max_length=157)
del X

np.save(data_path + 'finetuning/ids_atm.npy',  [input_ids, attention_masks])


token_train, token_test, mask_train, mask_test, \
y_train, y_test = train_test_split(np.array(input_ids),
                                   np.array(attention_masks),
                                   categories_labels,
                                   test_size=config.n_test,
                                   random_state=42)

np.save(data_path + 'finetuning/y_train.npy', y_train)
np.save(data_path + 'finetuning/y_test.npy', y_test)

##
# Supervised training (SciBert fine-tuning)

# Network architecture
input_ids = tf.keras.Input(shape=(config.input_size,), dtype='int32')
attention_masks = tf.keras.Input(shape=(config.input_size,), dtype='int32')
output = model([input_ids, attention_masks])
output = output[1]
output = tf.keras.layers.Dense(config.n_dense, activation='relu')(output)
output = tf.keras.layers.Dropout(config.dropout)(output)
output = tf.keras.layers.Dense(config.n_classes, activation='sigmoid')(output)
classification_model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

# Training
classification_model.summary()
classification_model.compile(Adam(learning_rate=config.lr), loss=config.criterion, metrics=['accuracy'])
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=data_path + 'finetuning/', monitor='val_accuracy', mode='max',
                                                 save_weights_only=True, save_best_only=False,
                                                 verbose=1)

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=config.patience, restore_best_weights=True)
history = classification_model.fit([token_train, mask_train], y_train, validation_split=config.n_validation,
                                   epochs=config.num_epochs,
                                   batch_size=config.batch_size, callbacks=[es, cp_callback])


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(data_path + 'img/tuning_results.png')

# Training embedding extraction
train_dataset = tf.data.Dataset.from_tensor_slices((token_train, mask_train))
train_dataset = train_dataset.batch(batch_size=500, drop_remainder=False)

train_tuned_bert_cls_embedding = np.array([])
train_tuned_bert_embedding = np.array([])
for id, atm in train_dataset:
    output = model.predict([id, atm])
    train_tuned_bert_embedding = np.concatenate([train_tuned_bert_embedding, output[0][:, 0, :]], 0) if train_tuned_bert_embedding.any() else output[0][:, 0, :]
    train_tuned_bert_cls_embedding = np.concatenate([train_tuned_bert_cls_embedding, output[1]], 0) if train_tuned_bert_cls_embedding.any() else output[1]


np.save(data_path + 'finetuning/tuned_scibert_train_embedding_from' + str(min_year) + '.npy', train_tuned_bert_embedding)
np.save(data_path + 'finetuning/tuned_scibert_cls_train_embedding_from' + str(min_year) + '.npy', train_tuned_bert_cls_embedding)


# Testing evaluation and embedding extraction
test_dataset = tf.data.Dataset.from_tensor_slices((token_test, mask_test))
test_dataset = test_dataset.batch(batch_size=500, drop_remainder=False)

test_tuned_bert_cls_embedding = np.array([])
test_tuned_bert_embedding = np.array([])
for id, atm in test_dataset:
    output = model.predict([id, atm])
    test_tuned_bert_embedding = np.concatenate([test_tuned_bert_embedding, output[0][:, 0, :]], 0) if test_tuned_bert_embedding.any() else output[0][:, 0, :]
    test_tuned_bert_cls_embedding = np.concatenate([test_tuned_bert_cls_embedding, output[1]], 0) if test_tuned_bert_cls_embedding.any() else output[1]


np.save(data_path + 'finetuning/tuned_scibert_test_embedding_from' + str(min_year) + '.npy', test_tuned_bert_embedding)
np.save(data_path + 'finetuning/tuned_scibert_cls_test_embedding_from' + str(min_year) + '.npy', test_tuned_bert_cls_embedding)

test_loss, test_accuracy = classification_model.evaluate([token_test, mask_test], y_test)
print("Testing loss:", test_loss, "Testing accuracy:", test_accuracy)

##
# Qualitative embedding evaluation

tsne = TSNE(random_state=42, init='pca')
train_tuned_bert_embedding_2D = tsne.fit_transform(train_tuned_bert_embedding)
train_tuned_bert_cls_embedding_2D = tsne.fit_transform(train_tuned_bert_cls_embedding)

test_tuned_bert_embedding_2D = tsne.fit_transform(test_tuned_bert_embedding)
test_tuned_bert_cls_embedding_2D = tsne.fit_transform(test_tuned_bert_cls_embedding)

# most frequent categories plot on the 2D-projection of the embedding

indices, cat_names = n_most_frequent_categories(categories_labels,  list(unique_categories_dict.keys()), 6)

# Train
plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(train_tuned_bert_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/train_tuned_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(train_tuned_bert_cls_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/train_tuned_cls_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

# Test
plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(test_tuned_bert_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/test_tuned_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(test_tuned_bert_cls_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/test_tuned_cls_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

# All

tuned_bert_embedding_2D = tsne.fit_transform(np.concatenate([train_tuned_bert_embedding, test_tuned_bert_embedding], 0))
tuned_bert_cls_embedding_2D = tsne.fit_transform(np.concatenate([train_tuned_bert_cls_embedding, test_tuned_bert_cls_embedding], 0))


plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(tuned_bert_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/tuned_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')

plt.figure(figsize=(25, 10))
for i in range(len(indices)):
    plt.subplot(2, 3, i + 1)
    scatter_one_class(tuned_bert_cls_embedding_2D, indices[i], cat_names[i])


plt.suptitle('TSNE with Subject Category Labels', fontsize=30)
plt.savefig(data_path + 'img/tuned_cls_Embedding_mostfrequentclass_from' + str(min_year) + '.png', bbox_inches='tight')
