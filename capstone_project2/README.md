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

## Model Limitations
It's crucial to acknowledge that the developed model, while showing promise, has limitations:

- Reliability Concerns: The model's predictions may not always be entirely accurate or reliable. It's essential to approach its outputs with caution and not solely rely on them for diagnostic decisions.

- Potential Misclassifications: Due to the complexity of Alzheimer's disease and the variability in MRI images, the model might misclassify certain cases, leading to false positives or false negatives.
Imbalanced image data presents challenges in model training:

- Class Distribution: The dataset might have significantly more samples from one class (e.g., non-demented) compared to others, impacting the model's ability to learn equally from each class.

- Biased Model Performance: The model might exhibit bias towards the majority class, affecting its ability to correctly classify minority classes (e.g., moderate demented).

## EDA/Model Building
These steps are included in[capstone2-alzheimer.ipynb](capstone2-alzheimer.ipynb)