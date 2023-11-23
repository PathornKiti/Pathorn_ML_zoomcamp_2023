from model import Model
import numpy as np
import tensorflow as tf

class slot_filling:
    def __init__(self):
        self.model = Model.load_bilstm()
        self.tokenizer_x,self.tokenizer_y = Model.load_bilstm_preprocessor()

    def fill_slot(self, text:str):
        label_list = list(self.tokenizer_y .word_index.keys())
        index_list = list(self.tokenizer_y .word_index.values())
        input_seq = self.tokenizer_x.texts_to_sequences([sentence])
        input_features = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=50, padding='post')

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        self.model.set_tensor(input_details[0]['index'], input_features.astype(np.float32))
        self.model.invoke()
        lstmpred = self.model.get_tensor(output_details[0]['index'])
        slots = [label_list[index_list.index(j)] for j in [np.argmax(x) for x in lstmpred[0][:]] if j in index_list]
        return slots
    