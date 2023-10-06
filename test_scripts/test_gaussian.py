from src.utils import *
from src.gaussian_classifier import GaussianClassifier


if __name__ == '__main__':
    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    gaussian_models = ["Multivariate", "Naive Bayes", "Tied", "Tied Naive Bayes"]
    classifiers = []
    for model in gaussian_models:
        gc = GaussianClassifier(model)
        classifiers.append(gc)

    leave_one_out_cross_validation(classifiers, data_matrix, labels)
