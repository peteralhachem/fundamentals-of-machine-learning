
from GMM import *


class GMMClassifier(GMM):
    def __init__(self, num_components, covariance_type):
        """

        :param num_components: Number of components used for the LBG algorithm to perform the split.
        :param covariance_type: Type of covariance used:[full, diagonal, tied].
        """
        super().__init__()
        self.labels = None
        self.num_components = num_components
        self.covariance_type = covariance_type
        self.class_gmm_components = None
        self.predicted_labels = None
        self.error = None

    def fit(self, data_matrix, labels):
        """
        Train the GMM classifier on the given data matrix and labels.
        :param data_matrix: Matrix to train (D,N) where D is the number of features and N is the number of samples.
        :param labels: The labels associated to each data point (N,) where N is the number of samples.

        """

        self.labels = labels
        self.class_gmm_components = []

        for label in np.unique(self.labels):
            self.Data = data_matrix[:, self.labels == label]
            self.gmm_components, log_likelihood = super().lbg_algorithm(self.Data, num_components=self.num_components,
                                                                        covariance_type=self.covariance_type)

            self.class_gmm_components.append(self.gmm_components)

    def predict(self, test_matrix, prior_probabilities=None):
        """
        Predict the labels for a given test matrix.
        :param test_matrix: The matrix used to calculate the accuracy/ error of the classifier.
        :param prior_probabilities: The probabilities of all the distinct classes of the dataset.
        :return predicted_labels: The labels that are predicted by our model of the dimension (N,).

        """

        self.Data = test_matrix
        class_joint_density = []
        class_marginal_density = []

        for label in np.unique(self.labels):

            self.gmm_components = self.class_gmm_components[label]
            class_joint_density.append(super()._gmm_log_density()[1])
            class_marginal_density.append(super()._gmm_log_density()[0])

        class_marginal_density = np.array(class_marginal_density)

        if prior_probabilities is None:

            prior_probabilities = [float(1.0/len(np.unique(self.labels)))] * len(np.unique(self.labels))

        log_priors = np.log(np.array(prior_probabilities))
        log_priors = log_priors.reshape((log_priors.size, 1))

        log_joint_distribution = class_marginal_density + log_priors

        self.predicted_labels = log_joint_distribution.argmax(axis=0)

        return self.predicted_labels

    def calculate_error(self, ground_truth):
        """

        Compute the error of the model where the error is the number of misclassified data points over all the data
        points.
        :param ground_truth: The true labels of the test dataset that we have used to predict the labels for.
        :return error:  an error rate that is represented in the percentual way.

        """

        bool_predictions = np.array(self.predicted_labels != ground_truth)

        self.error = (float(bool_predictions.sum() / ground_truth.shape[0])) * 100

        return self.error

    def __str__(self):
        """
        String function used to represent a way to display the values of a classifier and the corresponding error.
        :return:
        """
        string = f"Covariance Type = {self.covariance_type} | Number of Components = {self.num_components} | " \
                 f"error = {self.error}."

        string += "\n----------------------------------------------------\n"

        return string
