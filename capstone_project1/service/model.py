import pickle
import tensorflow as tf

class Model:
    def __init__(self) -> None:
        pass

    def load_cnn():
        interpreter_cnn = tf.lite.Interpreter(model_path="cnn.tflite")
        interpreter_cnn.allocate_tensors()
        return interpreter_cnn 

    def load_bilstm():
        interpreter_bilstm = tf.lite.Interpreter(model_path="bilstm.tflite")
        interpreter_bilstm.allocate_tensors()
        return interpreter_bilstm

    def load_cnn_preprocessor():
        with open('cnn_tokenizer.pickle', 'rb') as handle:
            cnn_tokenizer = pickle.load(handle)
        handle.close()
        with open('label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        handle.close()
        return cnn_tokenizer,le

    def load_bilstm_preprocessor():
        with open('lstmx_tokenizer.pickle', 'rb') as handle:
            lstmx_tokenizer= pickle.load(handle)
        handle.close()
        with open('lstmy_tokenizer.pickle', 'rb') as handle:
            lstmy_tokenizer= pickle.load(handle)
        handle.close()
        return lstmx_tokenizer,lstmy_tokenizer
        

        


