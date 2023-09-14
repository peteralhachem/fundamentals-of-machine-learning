
from Utils import *


class GaussianClassifier:
    def __init__(self, mode):
        self.mode = mode
        self.data = None
        self.labels = None
        self.mean = None
        self.covariance = None
        self.predicted_labels = None
        self.error = None
        self.accuracy = None
        self.is_error = False

    def fit(self, data_matrix, labels):

        """
        Train the Gaussian classifier on the given data matrix and labels, the training can be specified by the mode of
        covariance we have:[Tied, Multivariate, Naive Bayes, Tied Naive Bayes].
        :param data_matrix: Matrix to train (D,N) where D is the number of features and N is the number of samples.
        :param labels: The labels associated to each data point (N,) where N is the number of samples.

        """
        self.data = data_matrix
        self.labels = labels
        self.mean = calculate_class_means(self.data, self.labels)
        self.covariance = 0

        if self.mode == "Tied":
            for value in np.unique(self.labels):
                temp_value = calculate_covariance(center_data(self.data[:, self.labels == value]))
                self.covariance += (temp_value * (self.data[:, self.labels == value].shape[1]))

            self.covariance = self.covariance / self.data.shape[1]

        elif self.mode == "Multivariate":

            self.covariance = calculate_class_covariances(self.data, self.labels)

        elif self.mode == "Naive Bayes":

            self.covariance = calculate_class_covariances(self.data, self.labels)
            self.covariance = self.covariance * np.identity(self.data.shape[0])

        elif self.mode == "Tied Naive Bayes":

            for value in np.unique(self.labels):

                inter = calculate_covariance(center_data(self.data[:, self.labels == value]))
                self.covariance += (inter * (self.data[:, self.labels == value].shape[1]))

            self.covariance = self.covariance / self.data.shape[1]
            self.covariance = self.covariance * np.identity(self.data.shape[0])

    def predict(self, test_matrix, prior_probabilities=None):
        """
        Predict the labels for a given test matrix.
        :param test_matrix: The matrix used to calculate the accuracy/ error of the classifier.
        :param prior_probabilities: The probabilities of all the distinct classes of the dataset.
        :return predicted_labels: The labels that are predicted by our model of the dimension (N,).

        """
        if prior_probabilities is None:

            prior_probabilities = [float(1.0/len(np.unique(self.labels)))] * len(np.unique(self.labels))

        prior_probabilities = np.array(prior_probabilities)

        likelihood_scores = calculate_class_likelihood(self.mode, test_matrix, self.mean, self.covariance)

        joint_densities = likelihood_scores * prior_probabilities.reshape(prior_probabilities.size, 1)

        # ----Marginal is the summation of all the Joint densities of a sample within all the classes---- #

        marginal_densities = joint_densities.sum(0)
        marginal_densities = marginal_densities.reshape(1, marginal_densities.size)

        posterior_probabilities = joint_densities/marginal_densities

        self.predicted_labels = posterior_probabilities.argmax(axis=0)

        return self.predicted_labels

    def calculate_error(self, ground_truth):

        """
        Compute the error of the model where the error is the number of misclassified data points over all the data
        points.
        :param ground_truth: The true labels of the test dataset that we have used to predict the labels for.
        :return error:  an error rate that is represented in the percentual way.

        """
        self.is_error = True

        bool_predictions = np.array(self.predicted_labels != ground_truth)

        self.error = (float(bool_predictions.sum()/ground_truth.shape[0])) * 100

        return self.error

    def calculate_accuracy(self, ground_truth):
        """
        Compute the accuracy of the model where the accuracy is the number of well classified data points
        over all the data points.
        :param ground_truth: The true labels of the test dataset that we have used to predict the labels for.
        :return accuracy:  an accuracy rate that is represented in the percentual way.

        """

        boolean_predictions = np.array(self.predicted_labels == ground_truth)

        self.accuracy = float(boolean_predictions.sum() / ground_truth.shape[0]) * 100

        return self.accuracy

    def __str__(self):
        """
        String function used to represent a way to display the values of a classifier and
        the corresponding error/accuracy.
        :return:
        """

        if self.is_error:
            string = f"The error rate of the prediction of model {self.mode} : {self.error:.1f}%"
            string += "\n---------------------------------\n"

        else:
            string = f"The error rate of the prediction of model {self.mode} : {self.accuracy:.1f}%"
            string += "\n---------------------------------\n"

        return string
