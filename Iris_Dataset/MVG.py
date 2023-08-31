import numpy as np


def multivariate_gaussian(data_matrix, mu, cov):
    result = []

    for value in range(data_matrix.shape[1]):
        constant = -0.5 * cov.shape[0] * np.log(2 * np.pi)
        variable_1 = -0.5 * np.linalg.slogdet(cov)[1]

        variable_2 = data_matrix[:, value:value + 1] - mu
        variable_3 = -0.5 * np.dot(variable_2.T, np.dot(np.linalg.inv(cov), variable_2))

        result.append(constant + variable_1 + variable_3)

    return np.array(result).ravel()


def log_likelihood_estimator(data_matrix, mu, cov):
    result = multivariate_gaussian(data_matrix, mu, cov)
    return np.sum(result)
