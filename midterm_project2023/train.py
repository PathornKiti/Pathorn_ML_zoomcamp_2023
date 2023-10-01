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
df=pd.read_csv('credit_risk_dataset.csv')

numerical=df.select_dtypes(include=['number']).columns.tolist()
categorical= df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical.remove('loan_status')
    


# Data Preparation
X = df.drop('loan_status', axis=1)
X['person_emp_length'].fillna(X['person_emp_length'].mean(), inplace=True)
X['loan_int_rate'].fillna(X['loan_int_rate'].mean(), inplace=True)
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
    
    dicts = X_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model.fit(X_train, y_train, batch_size = 32, epochs = 50,verbose = 0)
    return dv,scaler,model



# Predict
def predict(df, dv,scaler,model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    X=scaler.fit_transform(X)
    y_pred = model.predict(X)

    return y_pred

dv,scaler,model=train(X_train, y_train)
print("Training finished")


y_pred=predict(X_test,dv,scaler,model)
y_pred = (y_pred > 0.5)
f1_test = f1_score(y_test, y_pred)
print(f"F1-score on Test Data: {f1_test}")

#Save model
with open('model.bin', 'wb') as f_out:
   pickle.dump((dv,scaler, model), f_out)
f_out.close()