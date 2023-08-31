
from Utils import *
import scipy


class MulticlassTaskEvaluator:

    def __init__(self, prior_vector, cost_matrix):
        self.prior_vector = prior_vector
        self.cost_matrix = cost_matrix

    def compute_dcf(self, class_log_likelihood, ground_truth):

        detection_cost = np.zeros(self.prior_vector.shape)

        joint_log_likelihood = class_log_likelihood + np.log(self.prior_vector.reshape(self.prior_vector.size, 1))
        marginal_log_likelihood = scipy.special.logsumexp(joint_log_likelihood, axis=0)

        posterior_probabilities = np.exp(joint_log_likelihood - marginal_log_likelihood)
        bayes_cost = np.dot(self.cost_matrix, posterior_probabilities)

        cost_of_dummy = np.min(np.dot(self.cost_matrix, self.prior_vector.reshape(self.prior_vector.size, 1)))
        predictions = np.argmin(bayes_cost, axis=0)

        confusion_matrix = compute_confusion_matrix(predictions, ground_truth)
        miss_classification_matrix = confusion_matrix / np.sum(confusion_matrix, axis=0)

        for index in range(self.prior_vector.shape[0]):

            detection_cost[index] = np.dot(np.dot(miss_classification_matrix[:, index], self.cost_matrix[:, index]),
                                           self.prior_vector[index])

        return detection_cost.sum(), detection_cost.sum() / cost_of_dummy
