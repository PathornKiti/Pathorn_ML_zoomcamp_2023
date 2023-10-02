import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np


    

#Load model
def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


preprocessor,model = load('model.bin')



app = Flask('loan-risk')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer_df = pd.DataFrame([customer])
    X=preprocessor.transform(customer_df)
    y_pred = model.predict(X)
    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred),
        'Default': bool(default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)