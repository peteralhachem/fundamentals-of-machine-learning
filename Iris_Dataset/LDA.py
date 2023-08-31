from Utils import *
from scipy.linalg import eigh

"""
U can either use the generalized method of to get the eigenvalues and eigenvectors or you can use the  joint 
diagonalization.

"""


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.between_covariance = 0
        self.within_covariance = 0
        self.W = None

    def fit(self, data_matrix, labels_matrix):

        for label in np.unique(labels):
            total_mean = calculate_mean(data_matrix[:, labels_matrix == label]) - calculate_mean(data_matrix)
            variable = np.dot(total_mean, total_mean.T) * data_matrix[:, labels_matrix == label].shape[1]
            self.between_covariance += variable

            new_value = data_matrix[:, labels_matrix == label] - calculate_mean(data_matrix[:, labels_matrix == label])
            within_covariance_value = calculate_covariance(new_value)

            self.within_covariance += (within_covariance_value * data_matrix[:, labels == label].shape[1])

        self.between_covariance = self.between_covariance / data_matrix.shape[1]
        self.within_covariance = self.within_covariance / data_matrix.shape[1]

        # Generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(self.between_covariance, self.within_covariance)

        self.W = eigenvectors[:, ::-1][:, 0:self.n_components]

    def transform(self, data_matrix):
        x_centered = center_data(data_matrix)

        x_transformed = np.dot(self.W.T, x_centered)

        return x_transformed


if __name__ == '__main__':
    data, labels = load_data_from_file("Dataset/iris.csv")

    lda = LDA(n_components=2)
    lda.fit(data, labels)
    transformed_matrix = lda.transform(data)

    scatter_plot(transformed_matrix, labels)
