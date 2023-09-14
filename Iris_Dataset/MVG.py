import numpy as np


def multivariate_gaussian(data_matrix, mu, cov):
    """
    Calculate the gaussian distribution of the data matrix considering each individual point seperated.
    :param data_matrix: Matrix to calculate the gaussian distribution (D,N) where D is the number of features and N
    is the number of samples.
    :param mu: The mean matrix computed for the corresponding data matrix with the shape (D,1).
    :param cov: The covariance matrix computed for the corresponding data matrix with the shape (D,D).
    :return: Joint distribution of the data matrix of the shape (N,) where N is the number of samples.

    """
    result = []

    for value in range(data_matrix.shape[1]):
        constant = -0.5 * cov.shape[0] * np.log(2 * np.pi)
        variable_1 = -0.5 * np.linalg.slogdet(cov)[1]

        variable_2 = data_matrix[:, value:value + 1] - mu
        variable_3 = -0.5 * np.dot(variable_2.T, np.dot(np.linalg.inv(cov), variable_2))

        result.append(constant + variable_1 + variable_3)

    return np.array(result).ravel()


def log_likelihood_estimator(data_matrix, mu, cov):
    """
    Compute the log likelihood for a given data matrix.
    :param data_matrix: Matrix to calculate the gaussian distribution (D,N) where D is the number of features and N
    is the number of samples.
    :param mu: The mean matrix computed for the corresponding data matrix with the shape (D,1).
    :param cov: The covariance matrix computed for the corresponding data matrix with the shape (D,D).
    :return: log-likelihood which is the summation of the log-joint density of all the data matrix.

    """

    result = multivariate_gaussian(data_matrix, mu, cov)
    return np.sum(result)
