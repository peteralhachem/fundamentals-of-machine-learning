
from Utils import *
from Gaussian_Classifiers import GaussianClassifier


if __name__ == '__main__':

    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    classifiers = []

    gc = GaussianClassifier("Multivariate")

    classifiers.append(gc)

    kfold_cross_validation(classifiers, data_matrix, labels, 9)
