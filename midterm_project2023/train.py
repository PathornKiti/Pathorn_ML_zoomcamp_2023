import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow import keras
import pickle

# Get the data
df=pd.read_csv('../pppppp/credit_risk_dataset.csv')

numerical=df.select_dtypes(include=['number']).columns.tolist()
categorical= df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical.remove('loan_status')

# Pipeline Preparation
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
    
#Define Preprocessing pipeline   
pipeline = Pipeline([
    ('numerical_imputer', NumericalImputer(variables=numerical)),
    ('categorical_encoder', DictVectorizerTransformer(variables=categorical)),
    ('scaler', StandardScaler())
])


# Data Preparation
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Create model
#Train model
def train(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', 
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    X_train=pipeline.fit_transform(X_train)
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model.fit(X_train, y_train, batch_size = 10, epochs = 50,verbose = 0)
    return pipeline,model

pipeline,model=train(X_train, y_train)
print("Training finished")

# Predict
def predict(df,pipeline,model):
    df=pipeline.fit_transform(df)
    y_pred = model.predict(df)
    return y_pred

y_pred=predict(X_test,pipeline,model)
y_pred = (y_pred > 0.5)
f1_test = f1_score(y_test, y_pred)
print(f"F1-score on Test Data: {f1_test}")

#Save model
with open('model.bin', 'wb') as f_out:
   pickle.dump((pipeline, model), f_out)
f_out.close()