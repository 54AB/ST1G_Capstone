# Use this command in terminal to start streamlit!
# streamlit run webAppSVMCancer.py

import streamlit as sl
from cancer import *
from PIL import Image



## streamlit page

# Title and sub title
sl.title("Cancer Cell Prediction")
sl.markdown("Predict if cell is cancerous based on its geometric features")

# Two tabs for data visualisation and prediction program
dsTab, ceTab = sl.tabs(["Data Showcase","Cell estimator"])

with dsTab:
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





