import streamlit as sl

from cancer import *

## streamlit page

# Title and sub title
sl.title("Cancer Cell Prediction")
sl.markdown("Predict if cell is cancerous based on its geometric features")

# Two tabs for data visualisation and prediction program
dsTab, ceTab = sl.tabs(
    [
        "Data Showcase",
        "Cell estimator"
    ]
)

with dsTab:
    sl.header("Cancer dataset")
    sl.write(cancer_dataset)

