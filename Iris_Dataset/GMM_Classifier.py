
from GMM import *


class GMMClassifier(GMM):
    def __init__(self, data_matrix, num_components, covariance_type):
        super().__init__(data_matrix)
        self.labels = None
        self.num_components = num_components
        self.covariance_type = covariance_type
        self.class_gmm_components = None
        self.predicted_labels = None
        self.error = None

    def fit(self, data_matrix, labels):

        self.labels = labels
        self.class_gmm_components = []

        for label in np.unique(self.labels):
            self.Data = data_matrix[:, self.labels == label]
            self.gmm_components, log_likelihood = super().lbg_algorithm(num_components=self.num_components,
                                                                        covariance_type=self.covariance_type)

            self.class_gmm_components.append(self.gmm_components)

        return self.class_gmm_components

    def predict(self, test_matrix, prior_probabilities=None):

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

        bool_predictions = np.array(self.predicted_labels != ground_truth)

        self.error = float(bool_predictions.sum() / ground_truth.shape[0]) * 100

        return self.error

    def __str__(self):
        string = "----------------------------------------------------\n"
        string += f"|\t {self.covariance_type} \t|\t {self.num_components} \t|\t {self.error:.1f}% \t|\n"

        return string
