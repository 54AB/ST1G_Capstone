
# Use Command (py -m pip install "package name") to install packages in VScode
# https://www.youtube.com/watch?v=ThU13tikHQw how to create virtual environment on VSCode
# How to chang execution policy if virtual environment cannot be initialisedhttps://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

## Load the Cancer Dataset
fileName = 'Cancer_Data.csv'
cancer_dataset = pd.read_csv(fileName)


def main():
    pass


## Exploratory data analysis

# Data head, Example of what the data looks like
print(cancer_dataset.head(5))

# shape
print(cancer_dataset.shape)

# Check data integrity (missing data)
cancer_dataset.info()

# Distribution of benign and malignant case
print(cancer_dataset.groupby('diagnosis').size())

#List of attributes
print(cancer_dataset.keys())

## Data Visualisation

#Drop unnecessary data entry for visualisaiton and re-examine the dataset
cancer_dataset.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
print(cancer_dataset.columns)
print(cancer_dataset.shape)

# Heat map
#Replace string B&M to binary 0, 1
cancer_dataset['diagnosis'] = cancer_dataset['diagnosis'].replace({'B': 0, 'M':1})
plt.subplots(figsize=(20, 20)) #Set size of the heat map
sns.heatmap(cancer_dataset[cancer_dataset.columns].corr(), annot=True)
plt.show()

# Distribution of each attribute
cancer_dataset.hist()
plt.show()

