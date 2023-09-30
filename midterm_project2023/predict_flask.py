import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Define custom class
class NumericalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        X=X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
    
class DictVectorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.dv = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        tmp_dict = X[self.variables + numerical].to_dict(orient='records')
        self.dv.fit(tmp_dict)
        return self

    def transform(self, X):
        tmp_dict = X[self.variables + numerical].to_dict(orient='records')
        return self.dv.transform(tmp_dict)
    
    
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


pipeline,model = load('model.bin')



app = Flask('loan-risk')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer= pd.DataFrame([customer])
    customer = pipeline.transform(customer)
    y_pred = model.predict(customer)[0][0]
    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred),
        'Default': bool(default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)