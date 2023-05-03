
# Use Command (py -m pip install "package name") to install packages in VScode
# https://www.youtube.com/watch?v=ThU13tikHQw how to create virtual environment on VSCode
# How to chang execution policy if virtual environment cannot be initialisedhttps://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
"""Accuracy: 0.951048951048951
Precision: 0.9525671896492566
Recall: 0.951048951048951
F1-score: 0.9505295489697844
[[88  1]
 [ 6 48]]
              precision    recall  f1-score   support

           0       0.94      0.99      0.96        89
           1       0.98      0.89      0.93        54

    accuracy                           0.95       143
   macro avg       0.96      0.94      0.95       143
weighted avg       0.95      0.95      0.95       143"""

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

# SVM with scaled data, proves that scaled data help the algorithm significantly 95% to 97%
"""Accuracy: 0.972027972027972
Precision: 0.972027972027972
Recall: 0.972027972027972
F1-score: 0.972027972027972
[[87  2]
 [ 2 52]]
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        89
           1       0.96      0.96      0.96        54

    accuracy                           0.97       143
   macro avg       0.97      0.97      0.97       143
weighted avg       0.97      0.97      0.97       143"""

## Using k-fold cross validation to test out different model and avoid over-fitting

#Model collection
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, scaled_X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Result
"""
KNN: 0.967220 (0.039206)
CART: 0.943798 (0.033072)
NB: 0.927076 (0.059788)
SVM: 0.978848 (0.024622)
"""

## Hyper parameter tuning for the best result

# hyper parameter of current better model
better_model = models[3][1]
params = model.get_params()
print(params)
# Current hyperprameter config
"""{'C': 1.0, 'break_ties': False, 'cache_size': 200,
 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr',
   'degree': 3, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1,
     'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001,
       'verbose': False}"""
params_grid = {'C':[10, 5, 1, 0.5, 0.1, 0.01], 'gamma':[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1], 'kernel': ['rbf']}
grs = GridSearchCV(model, params_grid, refit = True, verbose = 1, cv = 20)
grs.fit(scaled_X_train, y_train)

# Display hyper parameter changes
print ("\n Best Hyper Parameters: ", grs.best_params_,"\n", "Best estimator: ", grs.best_estimator_,"\n")

# Train model with hyper parameter tuned
model_best = grs.best_estimator_
y_prediction = model_best.predict(scaled_X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_prediction))
print("Precision:",metrics.precision_score(y_test, y_prediction, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_prediction, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_prediction, average = 'weighted'))
print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test, y_prediction))

