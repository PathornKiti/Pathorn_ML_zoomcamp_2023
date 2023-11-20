import pandas as pd
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     Dropout,Embedding,Bidirectional,TimeDistributed,LSTM)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical



df_train= pd.read_json('data/train.json')
X_train= df_train['utterance']
y_train= df_train['slots']
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=42)



print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)



x_tokenizer = Tokenizer(num_words=100000,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      oov_token = "<UNK>")

x_tokenizer.fit_on_texts(X_train)

X_train_token = x_tokenizer.texts_to_sequences(X_train)
X_valid_token = x_tokenizer.texts_to_sequences(X_valid)


y_tokenizer = Tokenizer(filters = '', lower = False, split = ' ')
y_tokenizer.fit_on_texts(y_train)
y_train_token = y_tokenizer.texts_to_sequences(y_train)
y_valid_token = y_tokenizer.texts_to_sequences(y_valid)

sequence_len = 50
X_train_token = pad_sequences(X_train_token, padding='post', maxlen=sequence_len)
X_valid_token = pad_sequences(X_valid_token, padding='post', maxlen=sequence_len)



y_train_token = pad_sequences(y_train_token, padding='post', maxlen=sequence_len)
y_valid_token = pad_sequences(y_valid_token, padding='post', maxlen=sequence_len)

y_train_encoded = to_categorical(y_train_token,num_classes=73)
y_valid_encoded = to_categorical(y_valid_token,num_classes=73)


X_train = np.reshape(X_train_token, (X_train_token.shape[0], X_train_token.shape[1], 1))
X_valid = np.reshape(X_valid_token, (X_valid_token.shape[0], X_valid_token.shape[1], 1))

vocab_size = len(x_tokenizer.index_word) + 1
y_vocab_size = len(y_tokenizer.index_word) + 1

bilstm = Sequential()
bilstm .add(Embedding(input_dim = vocab_size, output_dim = 64, input_length = sequence_len))
bilstm .add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
bilstm.add(Dropout(0.2))
bilstm .add(TimeDistributed(Dense(y_vocab_size, activation='softmax')))

bilstm .compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Precision(), Recall(), 'accuracy'])

checkpoint = ModelCheckpoint('lstm.h5', monitor='val_loss', save_best_only=True)
stop_early= tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history= bilstm.fit(X_train, y_train_encoded,
                    batch_size = 256,
                    epochs = 20,
                    validation_data=(X_valid, y_valid_encoded),
                    callbacks=[checkpoint,stop_early]
                    )

with open('x_tokenizer.pickle', 'wb') as handle:
    pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('y_tokenizer.pickle', 'wb') as handle:
    pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


converter_bilstm = tf.lite.TFLiteConverter.from_keras_model(bilstm)
converter_bilstm.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converter_bilstm._experimental_lower_tensor_list_ops = False

tflite_model_bilstm = converter_bilstm.convert()
open("bilstm.tflite", "wb").write(tflite_model_bilstm)