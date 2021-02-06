


"""
========================================================================================================================
======================= Classification applications on the handwritten digits data =====================================
========================================================================================================================
In this example, you will see two different applications of Naive Bayesian Algorithm on the
digits dataset.
"""

#import pylab as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
#import pylab as plt

########################################################################################################################
##################################### GETTING THE DATA & PREPARATIONS ##################################################
########################################################################################################################

np.random.seed(42)  # gets the same randomization each time
digits = load_digits()  # the whole dataset with the labels and other information are extracted
data = scale(digits.data)  # the data is scaled with the use of z-score
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels
print(digits.keys())  # this command will provide you the key elements in this dataset
print(digits.DESCR)  # to get the descriptive information about this dataset

########################################################################################################################
########################################################################################################################

from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#import pylab as plt
y = digits.target
X = digits.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

########################################################################################################################
########################################################################################################################

#TRAIN_TEST SPLIT

gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)
predicted = fit.predict(X_test)
print(confusion_matrix(y_test, predicted))
print("Accuracy:")
print(accuracy_score(y_test, predicted))  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("# of correct predictions:")
print(accuracy_score(y_test, predicted, normalize=False))  # the number of correct predictions
print("# of all predictions:")
print(len(predicted))  # number of all of the predictions

########################################################################################################################
########################################################################################################################

gnb = GaussianNB()
fit2 = gnb.fit(X, y)
predictedx = fit2.predict(X)
print(confusion_matrix(y, predictedx))
print("Accuracy:")
print(accuracy_score(y, predictedx))  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("# of correct predictions:")
print(accuracy_score(y, predictedx, normalize=False))  # the number of correct predictions
print("# of all predictions:")
print(len(predictedx))  # number of all of the predictions

unique_y, counts_y = np.unique(y, return_counts=True)
print(unique_y, counts_y)

unique_p, counts_p = np.unique(predictedx, return_counts=True)
print(unique_p, counts_p)
print((predictedx == 0).sum())
########################################################################################################################
########################################################################################################################