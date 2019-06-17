


# Breast Cancer Classification

## Introduction
 A small and compact Matlab script that solves a breast cancer classification with KNN, SVM, Naive Bayes and Decision Tree.
The dataset used is the breast-cancer-dataset-wisconsin ([http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)](http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)))
I have included some info about where this dataset has been used and which are the characteristics in dataSetInfo.txt.

The aim of the project was to have a first touch with solving classification problems on Matlab, so I have not spent much time optimizing the parameters or in general taking a lot of metric into account.

## Steps

* We import the .csv with our characteristics. We get rid of the first column which has the id and we create two matrixes. The first one has our data (10xnumbOfRows) and the second one our expected labels (1xnumbOfRows).

* We normalise the values of our data on the range [0, 1].
* We then perform KNN, SVM, Naive Bayes and Decision Tree achieving a max of 97% accuracy with correctly parametrised Naive Bayes (kfold cross validation with k = 10). The metrics used where: Accuracy, Sensitivity and Specificity.

