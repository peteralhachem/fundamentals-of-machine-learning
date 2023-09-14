from Utils import *


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.cov = None
        self.components = None

    def fit(self, data_matrix):
        """
        In the fitting section of PCA, we calculate the eigenvectors in ascending order of their eigenvalues.\n
        The eigenvectors are then chosen based on the number of components specified in the pca.

        :param data_matrix: matrix of data to perform PCA on.

        """

        centered_data = center_data(data_matrix)

        self.cov = calculate_covariance(centered_data)

        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)

        self.components = eigenvectors[:, ::-1][:, :self.n_components]

    def transform(self, data_matrix):
        """
        In the transform section of PCA, we perform a linear transformation on the new directions computed in the
        fit section.\n

        :param data_matrix: data matrix to perform transformation on.
        :return: Transformed data based on the new directions calculated by the eigenvectors.

        """
        centered_data = center_data(data_matrix)

        transformed_data = np.dot(self.components.T, centered_data)

        return transformed_data
