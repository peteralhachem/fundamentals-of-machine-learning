
from src.utils import *
import scipy


class MulticlassTaskEvaluator:

    def __init__(self, prior_vector, cost_matrix):
        """
        Constructor for the MulticlassTaskEvaluator class.
        :param prior_vector: vector of priors for all the classes available.
        :param cost_matrix: DxD matrix having a cost of mis-classification, the diagonal in this matrix is 0.

        """
        self.prior_vector = prior_vector
        self.cost_matrix = cost_matrix
        self.normalized_dcf = None
        self.dcf_u = None

    def compute_dcf(self, class_log_likelihood, ground_truth):
        """
        Compute the detection cost function of a multiclass dataset
        :param class_log_likelihood: Log likelihood of each datapoint in the dataset of each class.
        :param ground_truth: the true label of each datapoint in the dataset.
        :return: cost of all the distinct classes, comparing the cost of all the distinct classes with a cost of dummy.

        """

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

        self.dcf_u = detection_cost.sum()
        self.normalized_dcf = detection_cost.sum() / cost_of_dummy

        return self.dcf_u, self.normalized_dcf

    def __str__(self, eps=None):

        string = f"eps: {eps} | DCF_u: {self.dcf_u:.3f} | DCF: {self.normalized_dcf:.3f}"
        string += "\n-------------------------------------------------\n"

        return string

    def save_results(self, eps):
        """
        Save the results of the multiclass evaluator into a txt file.
        :param eps: epsilon parameter required for the evaluation.

        """

        if os.path.exists('../results/evaluation'):
            pass
        else:
            os.mkdir('../results/evaluation')

        if os.path.exists('../results/evaluation/multiclass_evaluation.txt'):
            with open('../results/evaluation/multiclass_evaluation.txt', 'a') as file:
                file.write(self.__str__(eps))
        else:
            try:
                with open('../results/evaluation/multiclass_evaluation.txt', 'a') as file:
                    file.write(self.__str__(eps))

            except FileNotFoundError:
                print('cannot create file.')
