import tensorflow as tf
import pandas as pd
import numpy as np
from os import path
from math import ceil
import json
import time
import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from model import NeuralTranslationModel

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=True,
	help="number of epochs")
ap.add_argument("-b", "--batch_size", required=True,
	help="batch size")
args = vars(ap.parse_args())

def map_splitting(english,chinese):
    return (tf.strings.split(english,' '),chinese)

def map_embedding(english,chinese):
    return (embedding_layer(english),chinese)

def filter_less_eq_max_token_len(english,chinese):
    return tf.less_equal(tf.shape(english)[0],tf.constant(max_len_in_chinese_tokenized))

def map_english_padding(english,chinese):
    english_length = tf.shape(english)[0]
    paddings = [[max_len_in_chinese_tokenized-english_length,0],
                [0,0]
               ]
    return (tf.pad(english,paddings=paddings),chinese)

df = pd.read_json('data/cmn-processed-tokenized.json')
embedding_layer = load_model('models/tf2-preview_nnlm-en-dim128_1')

tokenizer=[]
with open('data/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
tokenizer_config = tokenizer.get_config()
word_index = json.loads(tokenizer_config['word_index'])
max_word_index = max(word_index.values())

tokenizer_seq = df['chinese_tokenized']
max_len_in_chinese_tokenized = max([len(item) for item in tokenizer_seq])
chinese_padded_seq = pad_sequences(tokenizer_seq,maxlen = None,padding = "post")
x_train,x_test,y_train,y_test = train_test_split(df['english'].to_list(),
                                                 chinese_padded_seq,
                                                 train_size=0.90,
                                                 shuffle=True)

training_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))

training_dataset_split=training_dataset.map(map_splitting)
validation_dataset_split=validation_dataset.map(map_splitting)

training_dataset_filter = training_dataset_split.filter(filter_less_eq_max_token_len)
validation_dataset_filter = validation_dataset_split.filter(filter_less_eq_max_token_len)

count_train = training_dataset_filter.reduce(0, lambda x, _: x + 1)
count_val = validation_dataset_filter.reduce(0, lambda x, _: x + 1)

training_dataset_embed=training_dataset_filter.map(map_embedding)
validation_dataset_embed=validation_dataset_filter.map(map_embedding)

training_dataset_english_padded=training_dataset_embed.map(map_english_padding)
validation_dataset_english_padded=validation_dataset_embed.map(map_english_padding)

#training_dataset_shuffle = training_dataset_english_padded.shuffle(tf.cast(count_train,tf.int64),reshuffle_each_iteration=True)
#validation_dataset_shuffle= validation_dataset_english_padded.shuffle(tf.cast(count_val,tf.int64),reshuffle_each_iteration=True)

training_dataset_repeat = training_dataset_english_padded.repeat()
validation_dataset_repeat = validation_dataset_english_padded.repeat()

batch_size=int(args['batch_size'])
training_dataset_batch = training_dataset_repeat.batch(batch_size=batch_size)
validation_dataset_batch = validation_dataset_repeat.batch(batch_size=batch_size)

translation_model = NeuralTranslationModel(encoder_input_shape=(max_len_in_chinese_tokenized,128),
                                           decoder_input_shape=(max_word_index + 1, 128))

# define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
# define loss objective
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# compile the model
translation_model.compile(optimizer = optimizer,
                          loss = loss_object,
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# define callbacks
#checkpoint_epoch = ModelCheckpoint(filepath='models/eng-to-ch/checkpoint_epoch/checkpoint_{epoch}',
#                                   save_weights_only=True,
#                                   save_freq='epoch',
#                                   verbose=1)
checkpoint_best = ModelCheckpoint(filepath='models/eng-to-ch/checkpoint_best/checkpoint',
                                   save_weights_only=True,
                                   save_freq='epoch',
                                   save_best_only=True,
                                   monitor='val_sparse_categorical_accuracy',
                                   verbose=1)
lr_reduce_plateau = ReduceLROnPlateau(monitor='val_loss', 
                                      factor=0.05, 
                                      patience=5, 
                                      verbose=1, 
                                      mode='min')
early_stopping = EarlyStopping(monitor='val_loss', 
                               min_delta=0, 
                               patience=10, 
                               verbose=1,
                               mode='min')
callbacks=[checkpoint_best,lr_reduce_plateau,early_stopping]

steps_per_epoch = ceil(count_train/batch_size)
validation_steps = ceil(count_val/batch_size)

# fit the model
hist = translation_model.fit(training_dataset_batch,
                             epochs=int(args['epochs']),
                             steps_per_epoch=steps_per_epoch,
                             validation_data=validation_dataset_batch,
                             validation_steps=validation_steps,
                             callbacks=callbacks)

df_hist = pd.DataFrame(hist.history)
df_hist.to_json('data/training_hist.json')