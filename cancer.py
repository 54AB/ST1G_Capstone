
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

