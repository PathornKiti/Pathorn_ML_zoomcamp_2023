import pandas as pd
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     Dropout,Embedding,Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint



df_train= pd.read_json('data/train.json')
X_train= df_train['utterance']
y_train= df_train['intent']
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=42)


label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_valid= label_encoder.transform(y_valid)



print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)

tokenizer = Tokenizer(num_words=100000,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.index_word) + 1
X_train_token = tokenizer.texts_to_sequences(X_train)
X_valid_token = tokenizer.texts_to_sequences(X_valid)

sequence_len = 50
X_train_token = pad_sequences(X_train_token, padding='post', maxlen=sequence_len)
X_valid_token = pad_sequences(X_valid_token, padding='post', maxlen=sequence_len)

def build_cnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=sequence_len))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(units=len(np.unique(y_train)) , activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    return model
cnn = build_cnn_model()
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('cnn.h5', monitor='val_loss', save_best_only=True)
history = cnn.fit(X_train_token, y_train, epochs=20, batch_size=256,
                    validation_data=(X_valid_token , y_valid), callbacks=[checkpoint,stop_early])

with open('cnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
converter_cnn = tf.lite.TFLiteConverter.from_keras_model(cnn)
tflite_model_cnn = converter_cnn.convert()
open("cnn.tflite", "wb").write(tflite_model_cnn)
