# Loan risk prediction

This is my midterm project in machine learning Zoomcamp 2023. Please check out my code. Enjoy! :blush:

## Problem
The problem at hand is to develop a loan risk prediction system that assesses the creditworthiness of individuals applying for loans based on their personal information. Preventing loan defaults is crucial for financial institutions and lenders because defaults can have significant negative impacts on their financial health and overall operations. :money_with_wings:

## Goal
The goal is to determine whether an applicant is likely to default on the loan, providing financial institutions with a reliable tool to mitigate risk and make informed lending decisions. :chart_with_upwards_trend:

## Dataset
- [Loan risk data](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)

<img src="images/datadict.png" />

## EDA/Experiment/Model Selection
[Notebook](Loan_risk_pred.ipynb)
I finally choose ANN(Artificial Nueral Network) Classifier and I save my Preprocessing pipeline and the classifier model here. [Save model packages](model.bin)


## Run using python script
* If you are not sure about libraries please download [requirements.txt](requirements.txt) and then using this command `pip install -r requirements.txt` in command line
* Run this command `python train.py` in your terminal and make sure that your current directory path include this file [train.py](train.py). This script will write model.bin as an output
* After you got the model.bin. You can start to run `python predict.py` in your terminal to get customer default probability output. Make sure that your current directory path include this file [predict.py](predict.py). You can adjust customer feature within this python file


## Deployment with Flask
* You can run script [predict_flask.py](predict_flask.py) to get the endpoint. Then you can try open [test_flask_deployment.ipynb](test_flask_deployment.ipynb) to post a request to the endpoint