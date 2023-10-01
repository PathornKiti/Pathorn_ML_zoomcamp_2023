import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np


    
    
# Define features
numerical=['person_age',
 'person_income',
 'person_emp_length',
 'loan_amnt',
 'loan_int_rate',
 'loan_percent_income',
 'cb_person_cred_hist_length']
categorical=['person_home_ownership',
 'loan_intent',
 'loan_grade',
 'cb_person_default_on_file']




def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv,scaler,model = load('model.bin')



app = Flask('loan-risk')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    X=scaler.fit_transform(X)
    y_pred = model.predict(X)[0][0]
    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred),
        'Default': bool(default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)