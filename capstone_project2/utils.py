from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback,ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (Conv2D, 
                                     BatchNormalization, 
                                     MaxPooling2D, 
                                     Dropout, Flatten, Dense, input)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Sequential

def get_data(data_path):
    
    classes = [
        'MildDemented/',
        'ModerateDemented/',
        'NonDemented/',
        'VeryMildDemented/']

    file_paths = []
    labels = []

    for class_name in classes:
        folder = os.path.join(data_path, class_name)
        if os.path.isdir(folder):
            files = os.listdir(folder)
            file_paths.extend([os.path.join(folder, file) for file in files])
            labels.extend([class_name[:-1] for _ in range(len(files))])

    df = pd.DataFrame({'file_path': file_paths, 'label': labels})
    return df


def create_generator(batch_size,img_size,train_df,valid_df,test_df):

    train_datagen= ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
    test_datagen= ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
    # Keep mode rgb even the data are in grayscale to corresponse the pretrained model input channel
    train_gen = train_datagen.flow_from_dataframe(train_df, x_col= 'file_path', y_col= 'label', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

    valid_gen = test_datagen.flow_from_dataframe(valid_df, x_col= 'file_path', y_col= 'label', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

    test_gen = test_datagen.flow_from_dataframe(test_df, x_col= 'file_path', y_col= 'label', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)
    
    return train_gen,valid_gen,test_gen

def get_class_weight(df):
    class_labels = df['label'].unique()
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=df['label'])
    
    class_weights_dict = dict(zip(class_labels, class_weights))


    class_indices = {label: idx for idx, label in enumerate(class_labels)}


    class_weights_dict = {class_indices[label]: weight for label, weight in class_weights_dict.items()}

    return class_weights_dict
 
def create_callbacks(modelname,monitorvalue,mode):
    checkpoint = ModelCheckpoint(filepath=modelname,
                             monitor=monitorvalue,
                             save_best_only=True)

    stop = EarlyStopping(monitor=monitorvalue, 
                     mode=mode,
                     patience=15,
                     verbose=1, 
                     restore_best_weights=True)
    callbacks = [checkpoint,stop]
    
    return callbacks 

def build_pretrained(base_model,loss):

    METRICS = [
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),
    "accuracy"
    ]

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(l=0.01),
              activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006),
              activation='relu'),
        Dropout(rate= 0.4, seed= 42),
        Dense(4, activation='softmax')
        ])
    model.compile(loss=loss,
              optimizer=Adamax(learning_rate= 0.001),
              metrics=METRICS
                 )
    model.summary()
    return model

def convert_model(input_path,output_name):
    loaded_model = tf.keras.models.load_model(input_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()
    with open(output_name, 'wb') as f_out:
        f_out.write(tflite_model)