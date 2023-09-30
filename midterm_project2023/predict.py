import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import pickle


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


# Load model
with open('model.bin', 'rb') as f_in:
    pipeline,model = pickle.load(f_in)
f_in.close()


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

# Define customer
customer={'person_age': 24,
 'person_income': 15000,
 'person_home_ownership': 'RENT',
 'person_emp_length': 1.5,
 'loan_intent': 'VENTURE',
 'loan_grade': 'A',
 'loan_amnt': 10000,
 'loan_int_rate': 7.14,
 'loan_percent_income': 0.25,
 'cb_person_default_on_file': 'Y',
 'cb_person_cred_hist_length': 2}


customer= pd.DataFrame([customer])
customer = pipeline.transform(customer)

default_probability = model.predict(customer)[0][0]
print(f"Default Probability: {default_probability}")