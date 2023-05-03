
# Use Command (py -m pip install "package name") to install packages in VScode
# https://www.youtube.com/watch?v=ThU13tikHQw how to create virtual environment on VSCode
# How to chang execution policy if virtual environment cannot be initialisedhttps://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
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
plt.title("Correlation Matrix")
plt.show()

# Distribution of each attribute
cancer_dataset.hist()
plt.show()

## Preparing the data for analysis
X = cancer_dataset.drop('diagnosis', axis = 1) #Everything except the diagnosis column
y = cancer_dataset['diagnosis'] # The diagnosis column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


## Test run of analysis using Support vector machine
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Score for first attempt
"""Accuracy: 0.9473684210526315
Precision: 0.9514695830485304
Recall: 0.9473684210526315
F1-score: 0.9464615931721194
[[71  0]
 [ 6 37]]
              precision    recall  f1-score   support

           0       0.92      1.00      0.96        71
           1       1.00      0.86      0.92        43

    accuracy                           0.95       114
   macro avg       0.96      0.93      0.94       114
weighted avg       0.95      0.95      0.95       114"""

## More data pre-processing

# Feature scaling
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

model_with_scaled_data = SVC()
model_with_scaled_data.fit(scaled_X_train, y_train)
new_y_pred = model_with_scaled_data.predict(scaled_X_test)

print("Accuracy:",metrics.accuracy_score(y_test, new_y_pred))
print("Precision:",metrics.precision_score(y_test, new_y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, new_y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, new_y_pred, average = 'weighted'))
print(confusion_matrix(y_test, new_y_pred))
print(classification_report(y_test, new_y_pred))

# SVM with scaled data
"""Accuracy: 0.9824561403508771
Precision: 0.9829367940398942
Recall: 0.9824561403508771
F1-score: 0.9823691172375383
[[71  0]
 [ 2 41]]
              precision    recall  f1-score   support

           0       0.97      1.00      0.99        71
           1       1.00      0.95      0.98        43

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114"""


