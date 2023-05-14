import pandas as pd
import sys
import os
import ast
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
min_year = 2022
data = pd.read_csv(data_path + 'preprocessed_data_from' + str(min_year) + '.csv')
unique_categories_dict = np.load(data_path + 'unique_categories_dictionary_from' + str(min_year) + '.npy',
                                 allow_pickle=True)

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
        self.batch_size = 50
        self.num_epochs = 200  # maximum number of iterations
        self.lr = 6e-6  # learning rate
        self.patience = 3  # patience for early stopping
        self.criterion = 'categorical_crossentropy'  # loss
        self.n_validation = 0.2  # validation set proportion
        self.n_test = 0.1  # testing set proportion
        self.dropout = 0.2


config = Configuration()

####################
# TEXT EMBEDDING   #
# ##################

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = TFAutoModel.from_pretrained('allenai/scibert_scivocab_uncased', from_pt=True)

X = list(data['processed_abstract'])

input_ids, attention_masks = scibert_encode(X, tokenizer, max_length=157)
del X
np.save(data_path + 'categories_labels.npy',  categories_labels)
np.save(data_path + 'ids_atm.npy',  [input_ids, attention_masks])


token_train, token_test, mask_train, mask_test, \
y_train, y_test = train_test_split(np.array(input_ids),
                                   np.array(attention_masks),
                                   categories_labels,
                                   test_size=config.n_test,
                                   random_state=42)

##
test_dataset = tf.data.Dataset.from_tensor_slices((token_test, mask_test))
test_dataset = test_dataset.batch(batch_size=500, drop_remainder=False)

bert_cls_embedding = np.array([])
bert_embedding = np.array([])
for id, atm in test_dataset:
    output = model.predict([id, atm])
    bert_embedding = np.concatenate([bert_embedding, output[0][:, 0, :]], 0) if bert_embedding.any() else output[0][:, 0, :]
    bert_cls_embedding = np.concatenate([bert_cls_embedding, output[1]], 0) if bert_cls_embedding.any() else output[1]

np.save(data_path + 'scibert_test_embedding_from' + str(min_year) + '.npy', bert_embedding)
np.save(data_path + 'scibert_cls_test_embedding_from' + str(min_year) + '.npy', bert_cls_embedding)

##

input_ids = tf.keras.Input(shape=(config.input_size,), dtype='int32')
attention_masks = tf.keras.Input(shape=(config.input_size,), dtype='int32')
output = model([input_ids, attention_masks])
output = output[1]
output = tf.keras.layers.Dense(config.n_dense, activation='relu')(output)
output = tf.keras.layers.Dropout(config.dropout)(output)
output = tf.keras.layers.Dense(config.n_classes, activation='sigmoid')(output)
classification_model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

classification_model.summary()

classification_model.compile(Adam(learning_rate=config.lr), loss=config.criterion, metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=config.patience)

history = classification_model.fit([token_train, mask_train], y_train, validation_split=config.n_validation,
                                   epochs=config.num_epochs,
                                   batch_size=config.batch_size, callbacks=[es])


test_dataset = tf.data.Dataset.from_tensor_slices((token_test, mask_test))
test_dataset = test_dataset.batch(batch_size=500, drop_remainder=False)

tuned_bert_cls_embedding = np.array([])
tuned_bert_embedding = np.array([])
for id, atm in test_dataset:
    output = model.predict([id, atm])
    tuned_bert_embedding = np.concatenate([tuned_bert_embedding, output[0][:, 0, :]], 0) if tuned_bert_embedding.any() else output[0][:, 0, :]
    tuned_bert_cls_embedding = np.concatenate([tuned_bert_cls_embedding, output[1]], 0) if tuned_bert_cls_embedding.any() else output[1]


np.save(data_path + 'tuned_scibert_test_embedding_from' + str(min_year) + '.npy', tuned_bert_embedding)
np.save(data_path + 'tuned_scibert_cls_test_embedding_from' + str(min_year) + '.npy', tuned_bert_cls_embedding)

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(data_path + 'tuning_results.png')

test_loss, test_accuracy = classification_model.evaluate([token_test, mask_test], y_test)
print("Testing loss:", test_loss, "Testing accuracy:", test_accuracy)
