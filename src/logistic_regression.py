from scipy.optimize import fmin_l_bfgs_b
from utils import *


class LogisticRegression:
    def __init__(self, mode, value_lambda):
        """
        Construct LogisticRegression object.
        :param mode: type of LR can have the values: [Binary, Multiclass]
        :param value_lambda: value used for regularization can have the values:[10 ** -6, 1e-3, 1e-1, 1]

        """
        self.mode = mode
        self.value_lambda = value_lambda
        self.data_matrix = None
        self.labels = None
        self.W = None
        self.b = None
        self.J = None
        self.predicted_labels = None
        self.error = None

    def _logistic_regression_objective_function(self, v):
        """
        Objective function used for Binary Logistic Regression using it to calculate the gradient.
        :param v: N-dimensional array where N-1 values represent the W component and 1 value represent the b component.
        :return: empirical risk which is the summation of the regularization term and logistic loss.

        """

        w = v[0:-1]
        b = v[-1]
        log_value = 0

        for i in range(self.data_matrix.shape[1]):
            zi = 2 * self.labels[i] - 1
            log_value += np.logaddexp(0, -zi * (np.dot(w.T, self.data_matrix[:, i]) + b))

        logistic_loss = log_value / self.data_matrix.shape[1]
        regularization_term = 0.5 * self.value_lambda * (np.linalg.norm(w) ** 2)

        empirical_risk = regularization_term + logistic_loss

        return empirical_risk

    def _multiclass_logistic_regression_objective_function(self, v):

        """
        Objective function used for Multiclass Logistic Regression using it to calculate the gradient.
        :param v: N-dimensional array where N-dim values represent the W component
        and dim values represent the b component.
        :return: empirical risk which is the summation of the regularization term and logistic loss.

        """

        t_matrix = np.zeros((len(np.unique(self.labels)), self.data_matrix.shape[1]))

        for i in range(self.labels.shape[0]):
            for k in np.unique(self.labels):
                if self.labels[i] == k:
                    t_matrix[k, i] = 1

        dimension = len(np.unique(self.labels))
        self.W = v[:-dimension]
        self.b = v[-dimension:]

        self.W = self.W.reshape(self.data_matrix.shape[0], len(np.unique(self.labels)))
        self.b = self.b.reshape(len(np.unique(self.labels)), 1)

        score = np.dot(self.W.T, self.data_matrix) + self.b

        variable = np.log(np.sum(np.exp(score), axis=0))

        log_y = score - variable

        result = t_matrix * log_y

        final_value = np.sum(result) / self.data_matrix.shape[1]

        constant = (self.W * self.W).sum() * (0.5 * self.value_lambda)

        result = constant - final_value

        return result

    def fit(self, data_matrix, labels):

        """
        Train the LR classifier on the given data matrix and labels and retrieve values for W and b.
        Training the LR classifier could be done on a "Binary" dataset or on a "Multi-class" dataset.
        :param data_matrix: Matrix to train (D,N) where D is the number of features and N is the number of samples.
        :param labels: The labels associated to each data point (N,) where N is the number of samples.

        """

        self.data_matrix = data_matrix
        self.labels = labels

        if self.mode == "Binary":
            x0 = np.zeros(self.data_matrix.shape[0] + 1)
            x, self.J, _ = fmin_l_bfgs_b(self._logistic_regression_objective_function, x0,
                                         approx_grad=True,
                                         factr=10000000.0)
            self.W = x[:data_matrix.shape[0]]
            self.b = x[-1]

        elif self.mode == "Multiclass":
            x0 = np.zeros((self.data_matrix.shape[0] * len(np.unique(self.labels))) + len(np.unique(self.labels)))

            x, self.J, _ = fmin_l_bfgs_b(
                self._multiclass_logistic_regression_objective_function, x0,
                approx_grad=True,
                factr=10000000.0)

            self.W = x[:-len(np.unique(self.labels))]
            self.b = x[-len(np.unique(self.labels)):]

            self.W = self.W.reshape(self.data_matrix.shape[0], len(np.unique(self.labels)))
            self.b = self.b.reshape(len(np.unique(self.labels)), 1)

    def predict(self, test_matrix):

        """
        Predict the labels for a given test matrix.
        Contrary to a Gaussian classifier, LR scores do not have probabilistic meaning, and they are mainly based on
        linear transformed scores.
        :param test_matrix: The matrix used to calculate the accuracy/ error of the classifier.
        :return predicted_labels: The labels that are predicted by our model of the dimension (N,).

        """

        if self.mode == "Binary":
            score = np.zeros(test_matrix.shape[1])
            self.predicted_labels = np.zeros(test_matrix.shape[1])

            for i in range(test_matrix.shape[1]):
                score[i] = np.dot(self.W, test_matrix[:, i]) + self.b

            for i in range(test_matrix.shape[1]):
                if score[i] > 0:
                    self.predicted_labels[i] = 1
                else:
                    self.predicted_labels[i] = 0

        elif self.mode == "Multiclass":
            score = np.zeros((self.b.shape[0], test_matrix.shape[1]))
            self.predicted_labels = np.zeros(test_matrix.shape[1])

            for k in range(self.b.shape[0]):
                for i in range(test_matrix.shape[1]):
                    score[k, i] = np.dot(self.W[:, k], test_matrix[:, i]) + self.b[k]

            self.predicted_labels = np.argmax(score, axis=0)

        return self.predicted_labels

    def calculate_error(self, ground_truth):

        """
        Compute the error of the model where the error is the number of misclassified data points over all the data
        points.
        :param ground_truth: The true labels of the test dataset that we have used to predict the labels for.
        :return error:  an error rate that is represented in the percentual way.

        """

        boolean_values = np.array([ground_truth != self.predicted_labels])

        self.error = (boolean_values.sum() / ground_truth.shape[0]) * 100

        return self.error

    def __str__(self):

        """
        String function used to represent a way to display the values of a classifier and the corresponding error.
        :return:
        """

        string = f"Lambda:{self.value_lambda} | J:{self.J:.6e} | error:{self.error:.1f}%"
        string += "\n-----------------------------\n"

        return string
