from model import Model
import numpy as np
import tensorflow as tf

class intent_predict:
    def __init__(self):
        self.model = Model.load_cnn()
        self.tokenizer,self.le = Model.load_cnn_preprocessor()

    def get_intent(self, text:str):
    sentence_sequence = self.tokenizer.texts_to_sequences([text])
    sentence_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_sequence, 
                                                                             padding='post', maxlen=50)

    input_details = self.model.get_input_details()
    output_details = self.model.get_output_details()

    self.model.set_tensor(input_details[0]['index'], sentence_sequence_padded.astype(np.float32))
    self.model.invoke()
    cnnpred = self.model.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(cnnpred)
    label = self.le.classes_[predicted_label]
    return label