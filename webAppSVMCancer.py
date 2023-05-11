## Author : Zeguo Li
## Date Created: 26 April 2023
## Date Last changed 11 May 2023

## This program is a web app that uses cancer.py module to perfrom Machine learning estimation
#Content: Data showcase, Cell estimator
# Input: Cancer_data.csv from Cancer Data Kaggle, cancer.py artefacts (model, figures, ndarray etc.)
# Output: N/A
# This model utilise Cancer.py

# Use this command in terminal to start streamlit on VSCode!
# streamlit run webAppSVMCancer.py

### The Code ###

## Imports
# Use Command (py -m pip install "package name") to install packages in VScode.
import streamlit as sl
from cancer import raw_dataset, output, saveFile
from PIL import Image
import numpy as np
import pandas as pd
import joblib

## Load dataset
dataset = pd.read_csv('Cancer_Data.csv')
raw_dataset = dataset 

# While loop to load objects/model created by Cancer.py
# Will not stop until it is sure that everything is loaded
while True:
    try:
        loaded_model = joblib.load(saveFile)
        checkForNull = joblib.load('checkForNull')
        distBM = joblib.load('distBM')
        heat_map = joblib.load('heat_map')
        distplot = joblib.load('distplot')
        X = joblib.load('X')
        scaler = joblib.load('scaler')
        compare_algorithm = joblib.load('compare_algorithm')
        message = joblib.load('message')
        tuned_result = joblib.load('tuned_result')
        report = joblib.load('report')
        cancer_dataset = joblib.load('cancer_dataset_processed')
        break
    except FileNotFoundError:
        output(dataset) # First time running this code will be redirected here, cuz objects/model haven't been created yet.

## streamlit page

# Main page

def main():
    # Title and sub title
    sl.title("Cancer Cell Prediction")
    sl.markdown("Predict if cell is cancerous based on its geometric features")

    # Two tabs for data visualisation and prediction program
    dsTab, ceTab = sl.tabs(["Data Showcase","Cell estimator"])

    # Data Showcase Tab
    with dsTab:
        dataShowcase()

    # Cell Estimator Tab
    with ceTab:
        slider_input = create_sliders()
        make_prediction(slider_input)

# Prediction function

def make_prediction(sliderList):
    sliderList = np.array(list(sliderList)).reshape(1, -1) # Convert list to ndarray and lower dimmension
    sliderList = scaler.transform(sliderList) # Use scaler to scale original input, since the model is trained with scaled data
    predict = loaded_model.predict(sliderList) # Make prediction based on input

    if predict[0] == 0:
        sl.markdown("Result: Benign")
    else:
        sl.markdown("Result: Malignant")
        
    prediction_prob = loaded_model.predict_proba(sliderList) # Show confidence of model prediction
    proba = prediction_prob[0][predict[0]] # The prediction
    sl.metric(label = "Confidence", value = "{:.2f}%".format(proba*100),
                delta = "{:.2f}%".format((proba - 0.5) * 100)) # Show precentage change of confidence compare to indifference

# Create slider and record slider input

def create_sliders():
    sliderList = []
    for features in X.columns:  # Generate sliders using loop
        slider = sl.slider(label = features,
                            max_value = float(cancer_dataset[features].max()),
                            min_value = float(cancer_dataset[features].min()),
                            value = float(cancer_dataset[features].mean()))
        sliderList.append(slider)
    return sliderList

# Data visualisation and performance showcase

@sl.cache_data # Streamlit caching function, stores already loaded function into cache so no re-load is needed. Speed up the app.
def dataShowcase():
    sl.header("Cancer dataset")
    sl.write(raw_dataset)

    sl.subheader("Check for missing value")
    sl.write(checkForNull)

    sl.subheader("Number of benign and malignant cases in this dataset")
    sl.write(distBM)

    sl.header("Cancer dataset after cleaning")
    sl.text("Dropped ID and Blank column, convert diagnosis into binary data")
    sl.write(cancer_dataset)

    sl.header("Heat Map - Correlation Matrix")
    sl.pyplot(heat_map)

    sl.header("Distribution of dataset features and target")
    sl.pyplot(distplot)

    sl.header("Algorithm vs. Dataset")
    sl.text("Four algorithm was chosen for comparison.\nK Nearest Neighbour, Decision Tree, Naive Bayes, and Support Vector Machine.")
    sl.pyplot(compare_algorithm)
    sl.text(message)

    sl.header("Hyper parameter tuning result")
    sl.text(tuned_result)
    sl.subheader("Classification Report")
    sl.text(report)
    sl.subheader("Confusion Matrix of the model")
    image = Image.open('Confusion_Matrix.png')
    sl.image(image)

if __name__ == '__main__':
    main()






