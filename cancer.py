
# Use Command (py -m pip install "package name") to install packages in VScode
# https://www.youtube.com/watch?v=ThU13tikHQw how to create virtual environment on VSCode
# How to chang execution policy if virtual environment cannot be initialisedhttps://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
from lime import lime_tabular

## Load the Cancer Dataset
fileName = 'Cancer_Data.csv'
cancer_dataset = pd.read_csv(fileName)
raw_dataset = pd.read_csv(fileName)


# Classification report and Confusion matrix
def class_report_CM(y_test, y_pred):

    accuracy = "Accuracy: "+ str(metrics.accuracy_score(y_test, y_pred))
    precision = "Precision: "+ str(metrics.precision_score(y_test, y_pred, average = 'weighted'))
    recall = "Recall: "+ str(metrics.recall_score(y_test, y_pred, average = 'weighted'))
    f1Score = "F1-score: "+ str(metrics.f1_score(y_test, y_pred, average = 'weighted'))
    clrp = classification_report(y_test, y_pred)
    result = accuracy + "\n" + precision + "\n" + recall + "\n" + f1Score + "\n" + clrp
    print(result)
    print(confusion_matrix(y_test, y_pred))

    return result
    

## Exploratory data analysis
# Data explore - First five rows
first5Row = cancer_dataset.head(5)
print(first5Row)

# shape
print(cancer_dataset.shape)

# Check data integrity (missing data)
cancer_dataset.info()
checkForNull = cancer_dataset.isnull().sum()
print(checkForNull)

# Distribution of benign and malignant case
distBM = cancer_dataset.groupby('diagnosis').size()
print(distBM)

#List of attributes
print(cancer_dataset.keys())


## Data Visualisation


# Drop unnecessary data entry for visualisaiton and re-examine the dataset
cancer_dataset.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
# Check dataset after modification
print(cancer_dataset.columns)
print(cancer_dataset.shape)

# Replace string B&M to binary 0, 1
cancer_dataset['diagnosis'] = cancer_dataset['diagnosis'].replace({'B': 0, 'M':1})

# Heat map
heat_map, ax = plt.subplots(figsize=(20, 20)) #Set size of the heat map
sns.heatmap(cancer_dataset[cancer_dataset.columns].corr(), annot=True, ax = ax)
plt.title("Correlation Matrix")
plt.show()

# Distribution of each attribute
distplot = plt.figure(figsize = (28,21))
for feature in range(len(cancer_dataset.columns)):
    plt.subplot(6, 6, feature + 1)
    sns.histplot(data = cancer_dataset, x = cancer_dataset.columns[feature], hue = 'diagnosis' )

## Preparing the data for analysis
# Seperate feature and target set
X = cancer_dataset.drop('diagnosis', axis = 1) #Everything except the diagnosis column
y = cancer_dataset['diagnosis'] # The diagnosis column

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

model_with_scaled_data = SVC()
model_with_scaled_data.fit(scaled_X_train, y_train)
new_y_pred = model_with_scaled_data.predict(scaled_X_test)
class_report_CM(y_test, new_y_pred)


## Using k-fold cross validation to test out different model and avoid over-fitting

# Model collection - Spot Check
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
message = ""
for name, model in models:
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, scaled_X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    message += msg + '\n'
    print(msg)

# Compare Algorithms
compare_algorithm = plt.figure()
compare_algorithm.suptitle('Algorithm Comparison')
ax = compare_algorithm.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
    # return models

## Hyper parameter tuning for the best result

# def hypTuning(models, scaled_X_train, scaled_X_test, y_train, y_test):
# hyper parameter of current better model
better_model = models[3][1]
params = better_model.get_params()
print(params)
# Current hyperprameter config
"""{'C': 1.0, 'break_ties': False, 'cache_size': 200,
    'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr',
    'degree': 3, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': -1,
    'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001,
    'verbose': False}"""


# Since GridSearchCV() is a slow module to run, for the sake of speed,
# I will put the gridsearch result of the function in params_grid in input so it runs
# a lot more faster.
# The intended params_gird looks like this
# params_grid = {'C':[10, 5, 1, 0.5, 0.1, 0.01],
#                 'gamma':[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
#                 'kernel': ['rbf'],
#                 'probability': [True]}
# Output = Fitting 20 folds for each of 42 candidates, totalling 840 fits
# Best Hyper Parameters:  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True} 
# Best estimator:  SVC(C=10, gamma=0.01, probability=True)


# Find optimal hyper parameter for this model
params_grid = {'C':[10],
                'gamma':[0.01],
                'kernel': ['rbf'],
                'probability': [True]}
grs = GridSearchCV(better_model, params_grid, refit = True, verbose = 1, cv = 20)
grs.fit(scaled_X_train, y_train)

# Display hyper parameter changes
tuned_result = grs.best_estimator_
tune_result = "\n Best Hyper Parameters: ", grs.best_params_,"\n", "Best estimator: ", grs.best_estimator_,"\n"
print (tune_result)

# Train model with hyper parameter tuned
model_best = grs.best_estimator_
y_prediction = model_best.predict(scaled_X_test)
class_report_CM(y_test, y_prediction)
    # return model_best


## Save and load the trained model

saveFile = 'Best_SVM_Model.sav'
joblib.dump(model_best, saveFile)

# Load the model to make sure it is saved properly
loaded_model = joblib.load(saveFile)
final_y_pred = loaded_model.predict(scaled_X_test)
report = class_report_CM(y_test, final_y_pred)
cm = confusion_matrix(y_test, final_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.savefig('Confusion_Matrix.png')
plt.show()


# Performance interpretation using lime,
# with example of the third data in the test set


exp = lime_tabular.LimeTabularExplainer(scaled_X_train, 
                                        class_names= ['benign', 'maglinant'], 
                                        feature_names= X,
                                        mode = 'classification' )

explanation = exp.explain_instance(scaled_X_test[2], loaded_model.predict_proba,
                                num_features = len(X),
                                top_labels = 2)

explanation.as_pyplot_figure()
plt.show()
