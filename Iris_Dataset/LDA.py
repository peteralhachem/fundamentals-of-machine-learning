from Utils import *
from scipy.linalg import eigh


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.between_covariance = 0
        self.within_covariance = 0
        self.W = None

    def fit(self, data_matrix, labels_matrix):
        """
        In the fitting section of LDA, we calculate the eigenvectors in ascending order of their eigenvalues.\n
        The eigenvectors are then chosen based on the number of components specified in lda based on calculation made
        on the within and between covariance matrices.

        :param data_matrix: matrix of data to perform LDA on.
        :param labels_matrix: matrix of labels to perform LDA on.

        """
        for label in np.unique(labels_matrix):
            total_mean = calculate_mean(data_matrix[:, labels_matrix == label]) - calculate_mean(data_matrix)
            variable = np.dot(total_mean, total_mean.T) * data_matrix[:, labels_matrix == label].shape[1]
            self.between_covariance += variable

            new_value = data_matrix[:, labels_matrix == label] - calculate_mean(data_matrix[:, labels_matrix == label])
            within_covariance_value = calculate_covariance(new_value)

            self.within_covariance += (within_covariance_value * data_matrix[:, labels_matrix == label].shape[1])

        self.between_covariance = self.between_covariance / data_matrix.shape[1]
        self.within_covariance = self.within_covariance / data_matrix.shape[1]

        # ---Generalized eigenvalue problem ---#
        eigenvalues, eigenvectors = eigh(self.between_covariance, self.within_covariance)

        self.W = eigenvectors[:, ::-1][:, 0:self.n_components]

    def transform(self, data_matrix):
        """
        In the transform section of LDA, we perform a linear transformation on the new directions computed in the
        fit section.\n

        :param data_matrix: data matrix to perform transformation on.
        :return: Transformed data based on the new directions calculated by the eigenvectors.

        """
        x_centered = center_data(data_matrix)

        x_transformed = np.dot(self.W.T, x_centered)

        return x_transformed
