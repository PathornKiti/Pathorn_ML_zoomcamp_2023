## Dataset
The dataset consists of MRI images categorized into four classes:

- 1.Non-demented: Individuals without signs of dementia.
- 2.Very mild demented: Individuals with initial signs of cognitive decline.
- 3.Mild demented: Individuals with noticeable cognitive impairment.
- 4.Moderate demented: Individuals with substantial cognitive decline requiring assistance.

### Dataset Details
- **Number of Classes:** 4
- **Data Source:** [https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
API command to access the data  `kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images` 

## Problem Description
This project aims to develop a robust classification system for Alzheimer's disease using brain MRI images. Alzheimer's disease is a progressive neurodegenerative disorder that affects millions worldwide, primarily causing cognitive decline. Early detection and accurate classification of different stages of Alzheimer's disease are crucial for effective intervention and patient care.

## EDA/Model Building
These steps are included in[capstone2-alzheimer.ipynb](capstone2-alzheimer.ipynb)

## Solution
InceptionV3 is a convolutional neural network (CNN) architecture developed by Google. It is pre-trained on the ImageNet dataset and is known for its effectiveness in image recognition tasks. I leverage the InceptionV3 model to preprocess and predict Alzheimer's disease using MRI images.

## Deployment
Clone this repo and run the below docker command:
`Start Application:`
```docker
docker-compose up -d --build
```
and navigate to http://localhost:8501/
`Stop Application:`
```docker
docker-compose down
```
Dockerfile and environment configuration are included in backend and frontend folder. Models are deployed with FastAPI.
Streamlit deployment sample
<img src="pic/Screenshot 2567-01-09 at 23.25.57.png" />

## Model Limitations
It's crucial to acknowledge that the developed model, while showing promise, has limitations:

- **Supplement, Not Replacement**: However, it is crucial to understand that these models are supplementary and cannot replace the expertise and clinical judgment of trained medical professionals.

- **Human Expertise Required**: A doctor's diagnosis involves comprehensive analysis, considering various factors beyond image data, including patient history, symptoms, and other medical tests. Human expertise is crucial in interpreting results and making informed decisions regarding patient care.