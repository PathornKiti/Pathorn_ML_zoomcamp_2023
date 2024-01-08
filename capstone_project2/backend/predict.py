import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from typing import Any

interpreter = tflite.Interpreter(model_path="inceptionv3.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

preprocessor = create_preprocessor("inception_v3", target_size=(224, 224))

classes = [
    'MildDemented',
    'ModerateDemented',
    'NonDemented',
    'VeryMildDemented']

def run_classifier(image: str) -> Any:
    X = preprocessor.from_url(image)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    class_pred_dict = dict(zip(classes, preds[0]))
    result = max(class_pred_dict, key=class_pred_dict.get)
    return result