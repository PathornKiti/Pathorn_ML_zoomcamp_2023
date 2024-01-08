import tensorflow as tf
from utils import (get_data,create_generator,get_class_weight
                   ,create_callbacks,build_pretrained,convert_model)
from sklearn.model_selection import train_test_split
from tensorflow import keras



train_df=get_data("train_directory")
test_df=get_data("test_directory")
test_df, valid_df = train_test_split(test_df,  test_size= 0.6, shuffle= True, random_state= 123)

train_gen,valid_gen,test_gen=create_generator(16,
                                              (224,224),
                                              train_df,valid_df,test_df)

class_weights_dict=get_class_weight(train_df)

METRICS = [
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),
    "accuracy"
    ]

callbacks=create_callbacks("inceptionv3.h5","accuracy","max")
base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', 
                                                            include_top=False, 
                                                            input_shape=(224, 224, 3), 
                                                            pooling='max')
model=build_pretrained(base_model,tf.keras.losses.CategoricalFocalCrossentropy())
history = model.fit(train_gen, epochs=50, 
                    validation_data=valid_gen, 
                    class_weight=class_weights_dict,
                    callbacks=callbacks,
                    validation_steps= None, 
                    shuffle= False)

convert_model("/kaggle/working/inceptionv3.h5","inceptionv3.tflite")