from Utils import *
import scipy


class SVM:
    def __init__(self, mode, k, c, gamma=None, constant=None, degree=None):
        """
        Constructor for SVM.
        :param mode: Type of SVM used:[Linear, Polynomial, RBF].
        :param k: Value used to extend the matrix with: usually {1 or 10} for Linear and {0 or 1} for Kernel.
        :param c: Value used to bound the gradient when computed: usually with {0.1, 1, 10}.
        :param gamma: Value used to compute RBF: usually {1 or 10}.
        :param constant: Value used to compute Polynomial: usually {0 or 1}.
        :param degree: Value used to set the degree of the polynomial used: usually 2.

        """
        self.mode = mode
        self.data_matrix = None
        self.labels = None
        self.k = k
        self.c = c
        self.gamma = gamma
        self.constant = constant
        self.degree = degree
        self.z = None
        self.extended_data = None
        self.predicted_labels = None
        self.primal_loss = None
        self.dual_loss = None
        self.w_hat = None
        self.error = None

    def fit(self, data_matrix, labels):
        """
        Train the SVM classifier on the given data matrix and labels, the training can be specified by the mode of
        covariance we have:[Linear, Polynomial, RBF].
        :param data_matrix: Matrix to train (D,N) where D is the number of features and N is the number of samples.
        :param labels: The labels associated to each data point (N,) where N is the number of samples.

        """

        self.data_matrix = data_matrix
        self.labels = labels

    def _extend_matrix(self):
        """
        Extend the data matrix with the values of K.
        :return: extended data matrix.
        """

        row = np.tile(self.k, (1, self.data_matrix.shape[1]))

        self.extended_data = np.vstack((self.data_matrix, row))

        return self.extended_data

    @staticmethod
    def _polynomial(data_1, data_2, constant, degree, k):
        """
        Compute the polynomial function.
        :param data_1: First data matrix.
        :param data_2: Second data matrix.
        :param constant: value used to compute the polynomial.
        :param degree: degree of the polynomial.
        :param k: bias in the polynomial.
        :return: result of the polynomial function.
        """

        result = (np.dot(data_1.T, data_2) + constant) ** degree + k ** 2

        return result

    @staticmethod
    def _rbf(data_1, data_2, gamma, k):

        """
        Compute the RBF function.
        :param data_1: First data matrix.
        :param data_2: Second data matrix.
        :param gamma: constant used to calculate the RBF value.
        :param k: bias in the RBF.
        :return: result of the RBF function.
        """

        result = np.exp(-gamma * (np.linalg.norm(data_1 - data_2) ** 2)) + k ** 2

        return result

    def _calculate_h(self):
        """
        compute the h which is a variable needed to compute the objective function L(h).
        :return: h-variable of the objective function.
        """

        self.z = 2 * self.labels - 1

        self.Extended_Data = self._extend_matrix()

        if self.mode == "Linear":
            g = np.dot(self.extended_data.T, self.extended_data)

        elif self.mode == "Polynomial":
            g = self._polynomial(self.data_matrix, self.data_matrix, self.constant, self.degree, self.k)

        else:
            g = np.zeros((self.data_matrix.shape[1], self.data_matrix.shape[1]))
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    g[i, j] = self._rbf(self.data_matrix[:, i], self.data_matrix[:, j], self.gamma, self.k)

        h = np.dot(self.z.reshape(self.z.shape[0], 1), self.z.reshape(self.z.shape[0], 1).T) * g

        return h

    def _l_function(self, alpha):
        """
        Dual Objective function under the name L(alpha).
        Our objective is to take this function and minimize its value.
        :param alpha: the argument of the objective function that will let us find the primal loss and the linear
        transformation vector.
        :return:
        """

        h = self._calculate_h()
        ones = np.ones(self.extended_data.shape[1])

        result = 0.5 * np.dot(np.dot(alpha.T, h), alpha) - np.dot(alpha.T, ones)

        gradient = np.dot(h, alpha) - ones

        return result, gradient.reshape(gradient.size)

    def _find_alpha(self):
        """
        Calculate the value of alpha by using the objective function L(alpha) and passing an initial value and bounds.
        Using a gradient descent algorithm we are able to get the value of the vector X as well as the primal loss.
        :return:
        """

        self.extended_data = self._extend_matrix()

        bound = [(0, self.c)] * self.Extended_Data.shape[1]
        x0 = np.zeros(self.Extended_Data.shape[1])

        x, self.Primal_loss, _ = scipy.optimize.fmin_l_bfgs_b(self._l_function, x0=x0, bounds=bound, factr=1.0)

        return x, self.primal_loss

    def predict(self, test_matrix):

        """
        Predict the labels for a given test matrix.
        Contrary to a Gaussian classifier, SVM scores do not have probabilistic meaning, and they are mainly based on
        linear transformed scores.
        :param test_matrix: The matrix used to calculate the accuracy/ error of the classifier.
        :return predicted_labels: The labels that are predicted by our model of the dimension (N,).

         """

        score = np.zeros(test_matrix.shape[1])

        alpha, self.primal_loss = self._find_alpha()

        if self.mode == "Linear":

            self.w_hat = np.dot(self.z * self.extended_data, alpha)
            w = self.w_hat[:test_matrix.shape[0]].reshape(self.w_hat[:test_matrix.shape[0]].shape[0], 1)
            b = self.w_hat[test_matrix.shape[0]]

            for i in range(test_matrix.shape[1]):
                score[i] = np.dot(w.T, test_matrix[:, i]) + b

        elif self.mode == "Polynomial":
            score = np.dot(alpha * self.z, self._polynomial(self.data_matrix, test_matrix, self.constant, self.degree,
                                                            self.k))
        elif self.mode == "RBF":

            kernel_values = np.zeros((self.data_matrix.shape[1], test_matrix.shape[1]))

            for i in range(self.data_matrix.shape[1]):
                for j in range(test_matrix.shape[1]):
                    kernel_values[i, j] = self._rbf(self.data_matrix[:, i], test_matrix[:, j], self.gamma, self.k)

            score = np.dot(alpha * self.z, kernel_values)

        self.predicted_labels = np.int32(score > 0)

        return self.predicted_labels

    def _loss_for_linear(self):
        """
        Compute the dual loss, primal loss and dual gap for the linear SVM.
        :return:
        """

        maxsum = 0

        for i in range(self.Extended_Data.shape[1]):
            maxsum += max(0, 1 - (self.z[i] * np.dot(self.w_hat, self.extended_data[:, i])))

        self.dual_loss = 0.5 * (np.linalg.norm(self.w_hat)) ** 2 + self.c * maxsum

        dual_gap = self.dual_loss + self.Primal_loss

        return self.dual_loss, self.Primal_loss, dual_gap

    def _loss_for_kernel(self):
        """
        Compute the dual loss for the given kernel SVM.
        :return:
        """

        self.dual_loss = -self.primal_loss

        return self.dual_loss

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

        string = ""

        if self.mode == 'Linear':
            self.dual_loss, self.primal_loss, dual_gap = self._loss_for_linear()
            string = f"K={self.k} | C={self.c} | primal loss={-self.primal_loss:.6e} | dual loss={self.dual_loss:.6e}" \
                     f"| dual gap={dual_gap:6e} | error = {self.error:.1f}%\n "
            string += "\n-----------------------------\n"

        elif self.mode == 'Polynomial':
            self.dual_loss = self._loss_for_kernel()
            string = f"Kernel:{self.mode} | K={self.k} | C={self.c} | (d={self.degree},c={self.constant}) |" \
                     f" dual loss={self.dual_loss:.6e} | error = {self.error:.1f}%\n "
            string += "\n-----------------------------\n"

        elif self.mode == 'RBF':
            self.dual_loss = self._loss_for_kernel()
            string = f"Kernel:{self.mode} | K={self.k} | C={self.c} | (gamma={self.gamma}) | dual loss=" \
                     f"{self.dual_loss:.6e} | error = {self.error:.1f}%\n "
            string += "\n-----------------------------\n"

        return string
