# Natural Language Understanding (NLU)
This project focuses on Natural Language Understanding (NLU), a critical component of conversational AI systems. The primary goal is to develop and train models that can accurately interpret user input (utterances) to extract intents and identify specific entities or slots within the text.

## Problem Statement
The problem revolves around the challenge of understanding user queries or statements in natural language. It involves:

- Extracting the intention behind user input: Given diverse user utterances, the system needs to accurately identify the user's intent or purpose, such as querying weather information, making reservations, playing music, etc.
- Recognizing entities or slots: Understanding and categorizing specific entities within the user input, such as dates, locations, names, quantities, etc., using techniques like Named Entity Recognition (NER) with the BIO tagging scheme (Beginning, Inside, Outside).

<img src="pic/example.jpg" />
Reference:https://www.researchgate.net/figure/shows-an-example-utterance-from-the-Snips-dataset-Find-a-novel-called-industry-with-a_fig1_337405537

## Dataset
The dataset (train.json,valid.json) contains examples of user utterances along with associated intents and slot annotations, serving as the foundation for model training and evaluation.
Reference:https://www.kaggle.com/competitions/hackathon-online-nlu-intent-classification/overview