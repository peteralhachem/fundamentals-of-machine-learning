from src.utils import *
from scipy.linalg import eigh


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.between_covariance = 0
        self.within_covariance = 0
        self.W = None
        self.data_matrix = None
        self.transformed_data = None

    def fit(self, data_matrix, labels_matrix):
        """
        In the fitting section of LDA, we calculate the eigenvectors in ascending order of their eigenvalues.\n
        The eigenvectors are then chosen based on the number of components specified in lda based on calculation made
        on the within and between covariance matrices.

        :param data_matrix: matrix of data to perform LDA on.
        :param labels_matrix: matrix of labels to perform LDA on.

        """
        self.data_matrix = data_matrix

        for label in np.unique(labels_matrix):
            total_mean = calculate_mean(data_matrix[:, labels_matrix == label]) - calculate_mean(self.data_matrix)
            variable = np.dot(total_mean, total_mean.T) * self.data_matrix[:, labels_matrix == label].shape[1]
            self.between_covariance += variable

            new_value = self.data_matrix[:, labels_matrix == label] - calculate_mean(
                self.data_matrix[:, labels_matrix == label])
            within_covariance_value = calculate_covariance(new_value)

            self.within_covariance += (within_covariance_value * self.data_matrix[:, labels_matrix == label].shape[1])

        self.between_covariance = self.between_covariance / self.data_matrix.shape[1]
        self.within_covariance = self.within_covariance / self.data_matrix.shape[1]

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

        self.transformed_data = np.dot(self.W.T, x_centered)

        return self.transformed_data

    def __str__(self):
        string = f"PCA with n={self.n_components}, initial data shape={self.data_matrix.shape} ---> reduced data shape"\
                 f"={self.transformed_data.shape}."

        return string

    def save_results(self):
        """
        Save the results of the lda class.

        """

        # check if directory exists
        if os.path.exists("../results/lda"):
            pass
        else:
            os.mkdir("../results/lda")

        if os.path.exists('../results/lda/lda_%d.txt' % self.n_components):
            with open('../results/lda/lda_%d.txt' % self.n_components, 'w') as file:
                file.write(self.__str__())

        else:
            try:
                with open('../results/lda/lda_%d.txt' % self.n_components, 'w') as file:
                    file.write(self.__str__())

            except FileNotFoundError:
                print("Could not create file.")
