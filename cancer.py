## Author : Zeguo Li
## Date Created: 25 April 2023
## Date Last changed 11 May 2023

## This program is an attempt to create a machine learning model that predicts the state of tumor cell using dataset Cancer - Kaggle
# Target Variable: B - Benign, M - Malignant
# Features number: 30 (Radius, Smoothness, Concavity etc.)
# Input: Cancer_data.csv from Cancer Data Kaggle.
# Output: Pre-trained SVM model
# This module will be imported by a Web App created by streamlit


### Help I got from the internet when creating virtual environment ###

# Gow to create virtual environment on VSCode. 
# https://www.youtube.com/watch?v=ThU13tikHQw 
# How to change execution policy if virtual environment cannot be initialised.
# https://stackoverflow.com/questions/18713086/virtualenv-wont-activate-on-windows

### The Code ###

# Imports
# Use Command (py -m pip install "package name") to install packages in VScode.

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

# Model save name

saveFile = 'Best_SVM_Model.sav'

## Load the Cancer Dataset

fileName = 'Cancer_Data.csv'
cancer_dataset = pd.read_csv(fileName)
raw_dataset = pd.read_csv(fileName)

## Function for the Streamlit Web application

def output(brand_new_dataset):
    checkForNull, distBM, drop_dataset = EDA(brand_new_dataset)
    heat_map, distplot, dataset = dataVisualisation(drop_dataset)
    X, scaled_X_train, scaled_X_test, y_train, y_test, scaler = dataPreperation(dataset)
    models, fig, message = modelComparison(scaled_X_train, y_train)
    model_best, tuned_result, report = hypTuning(models, scaled_X_train, scaled_X_test, y_train, y_test)
    modelInterpretation(scaled_X_train, scaled_X_test, model_best, X)
    joblib.dump(checkForNull, 'checkForNull')
    joblib.dump(distBM, 'distBM')
    joblib.dump(heat_map, 'heat_map')
    joblib.dump(distplot, 'distplot')
    joblib.dump(X, 'X')
    joblib.dump(scaler,'scaler')
    joblib.dump(fig, 'compare_algorithm')
    joblib.dump(message, 'message')
    joblib.dump(tuned_result, 'tuned_result')
    joblib.dump(report, 'report')
    joblib.dump(dataset, 'cancer_dataset_processed')

## Main function

def main():
    drop_dataset = EDA(cancer_dataset)[2]
    dataset = dataVisualisation(drop_dataset)[2]
    X, scaled_X_train, scaled_X_test, y_train, y_test, scaler = dataPreperation(dataset)
    models, fig, message = modelComparison(scaled_X_train, y_train)
    model_best, tuned_result, report = hypTuning(models, scaled_X_train, scaled_X_test, y_train, y_test)
    modelInterpretation(scaled_X_train, scaled_X_test, model_best, X)



## Function for terminal Classification report and Confusion matrix

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

def EDA(dataset):
    # Data explore - First five rows
    dataset.head(5)

    # shape
    print(dataset.shape)

    # Check data integrity (missing data)
    dataset.info()
    checkForNull = dataset.isnull().sum()
    print(checkForNull)

    # Distribution of benign and malignant case
    distBM = dataset.groupby('diagnosis').size()
    print(distBM)

    #List of attributes
    print(dataset.keys())

    # Drop unnecessary data entry for visualisaiton and re-examine the dataset
    dataset.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
    # Check dataset after modification
    print(dataset.columns)
    print(dataset.shape)

    return checkForNull, distBM, dataset


## Data Visualisation

def dataVisualisation(dataset):

    # Replace string B&M to binary 0, 1 so that correlation can be analysed
    dataset['diagnosis'] = dataset['diagnosis'].replace({'B': 0, 'M':1})

    # Heat map
    heat_map, ax = plt.subplots(figsize=(20, 20)) #Set size of the heat map
    sns.heatmap(dataset[dataset.columns].corr(), annot=True, ax = ax)
    plt.title("Correlation Matrix")
    plt.show()

    # Distribution of each attribute
    distplot = plt.figure(figsize = (28,21))
    for feature in range(len(dataset.columns)):
        plt.subplot(6, 6, feature + 1)
        sns.histplot(data = dataset, x = dataset.columns[feature], hue = 'diagnosis' )
    
    return heat_map, distplot, dataset

## Preparing the data for analysis

def dataPreperation(dataset):

    # Seperate feature and target set and Feature scaling
    scaler = StandardScaler()
    X = dataset.drop('diagnosis', axis = 1) #Everything except the diagnosis column
    X_scaled = scaler.fit_transform(X)
    y = dataset['diagnosis'] # The diagnosis column

    # Train test split
    scaled_X_train, scaled_X_test, y_train, y_test = train_test_split(
                                        X_scaled, y, test_size = 0.2, random_state = 42)
    return X, scaled_X_train, scaled_X_test, y_train, y_test, scaler


## Using k-fold cross validation to test out different model and avoid over-fitting

def modelComparison(scaled_X_train, y_train):
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
    return models, compare_algorithm, message

## Hyper parameter tuning for the best result

def hypTuning(models, scaled_X_train, scaled_X_test, y_train, y_test):
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


    """Since GridSearchCV() is a slow function to run, for the sake of speed,
    I will put the gridsearch result of the function in params_grid in input so it runs
    a lot more faster.
    The intended params_gird looks like this
    params_grid = {'C':[10, 5, 1, 0.5, 0.1, 0.01],
                    'gamma':[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
                    'kernel': ['rbf'],
                    'probability': [True]}
    Output = Fitting 20 folds for each of 42 candidates, totalling 840 fits
    Best Hyper Parameters:  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True} 
    Best estimator:  SVC(C=10, gamma=0.01, probability=True)"""

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

    # Generate report and confusion matrix for web app
    report = class_report_CM(y_test, y_prediction)
    cm = confusion_matrix(y_test, y_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.savefig('Confusion_Matrix.png')
    plt.show()
    # Save the model for later use in Web App
    joblib.dump(model_best, saveFile)

    return model_best, tuned_result, report

## Performance interpretation using lime,
# with example of the third data in the test set
def modelInterpretation(scaled_X_train, scaled_X_test, model_best, X):

    exp = lime_tabular.LimeTabularExplainer(scaled_X_train, 
                                            class_names= ['benign', 'maglinant'], 
                                            feature_names= X,
                                            mode = 'classification' )

    explanation = exp.explain_instance(scaled_X_test[2], model_best.predict_proba,
                                    num_features = len(X),
                                    top_labels = 2)

    explanation.as_pyplot_figure()
    plt.show()

## Protect the code from runniing in case of import
if __name__ == '__main__':
    main()