import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import pickle



# Load model
with open('model.bin', 'rb') as f_in:
    preprocessor,model = pickle.load(f_in)
f_in.close()



# Define customer
customer={'person_age': 26,
 'person_income': 43200,
 'person_home_ownership': 'OWN',
 'person_emp_length': 1.0,
 'loan_intent': 'EDUCATION',
 'loan_grade': 'C',
 'loan_amnt': 17000,
 'loan_int_rate': 13.49,
 'loan_percent_income': 0.39,
 'cb_person_default_on_file': 'Y',
 'cb_person_cred_hist_length': 2}


def predict(customer):
    customer_df = pd.DataFrame([customer])
    X=preprocessor.transform(customer_df)
    y_pred = model.predict(X)
    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred),
        'Default': bool(default)
    }
    return result

pred=predict(customer)

print(f"{pred}")