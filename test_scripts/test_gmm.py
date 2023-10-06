from src.utils import *
from src.gmm_classifier import GMMClassifier

if __name__ == '__main__':

    covariance_types = ['full', 'diagonal', 'tied']
    num_components = [1, 2, 4, 8, 16]

    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    for cov_type in covariance_types:
        for num in num_components:
            gmm_c = GMMClassifier(num, cov_type)
            gmm_c.fit(DTR, LTR)
            gmm_c.predict(DTE)
            gmm_c.calculate_error(LTE)
            gmm_c.save_results()
