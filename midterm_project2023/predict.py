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
    dv,scaler,model = pickle.load(f_in)
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
customer={"person_age":24,
          "person_income":54400,
          "person_home_ownership":"RENT",
          "person_emp_length":8.0,
          "loan_intent":"MEDICAL",
          "loan_grade":"C",
          "loan_amnt":35000,
          "loan_int_rate":14.27
          ,"loan_percent_income":0.55,
          "cb_person_default_on_file":"Y",
          "cb_person_cred_hist_length":4}


def predict(customer):
    X = dv.transform([customer])
    X=scaler.fit_transform(X)
    y_pred = model.predict(X)[0][0]
    default = y_pred >= 0.5

    result = {
        'Default_probability': float(y_pred),
        'Default': bool(default)
    }
    return result

pred=predict(customer)

print(f"{pred}")