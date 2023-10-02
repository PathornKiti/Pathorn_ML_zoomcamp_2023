import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
import pickle

# Get the data
df=pd.read_csv('credit_risk_dataset.csv')
df.drop(df[df['person_age'] > 80].index, axis=0,inplace=True)
df.drop(df[df['person_emp_length'] > 65].index, axis=0,inplace=True)

numerical=df.select_dtypes(include=['number']).columns.tolist()
categorical= df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical.remove('loan_status')



#Preprocessor Build
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
preprocessor = ColumnTransformer(
    transformers=[
        #('feature_engineering', fea_eng, numerical),
        ('numeric_transformers', numeric_transformer, numerical),
        ('categorical_transformers', cat_transformer, categorical),
    ])





# Data Preparation
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Create model
#Train model
def train(X_train, y_train):
    X_train=preprocessor.fit_transform(X_train)
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    model = xgb.XGBClassifier(
    max_depth=7,
    learning_rate=0.01,
    subsample=0.7,)
    model.fit(X_train, y_train)
    return preprocessor,model



# Predict
def predict(df, preprocessor,model):
    X = preprocessor.transform(df)
    y_pred = model.predict(X)

    return y_pred

preprocessor,model=train(X_train, y_train)
print("Training finished")


y_pred=predict(X_test, preprocessor,model)
y_pred = (y_pred > 0.5)
f1_test = f1_score(y_test, y_pred)
print(f"F1-score on Test Data: {f1_test}")
target_names = ['non-default', 'default']
print(classification_report(y_test, y_pred,target_names=target_names))

#Save model
with open('model.bin', 'wb') as f_out:
   pickle.dump((preprocessor,model), f_out)
f_out.close()