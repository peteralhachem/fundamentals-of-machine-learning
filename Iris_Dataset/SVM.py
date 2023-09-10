from Utils import *
import scipy


class SVM:
    def __init__(self, mode):
        self.mode = mode
        self.data_matrix = None
        self.labels = None
        self.k = None
        self.c = None
        self.gamma = None
        self.constant = None
        self.degree = None
        self.z = None
        self.extended_data = None
        self.predicted_labels = None
        self.primal_loss = None
        self.dual_loss = None
        self.w_hat = None

    def fit(self, data_matrix, labels, k, c, gamma=None, constant=None, degree=None):
        self.data_matrix = data_matrix
        self.labels = labels
        self.k = k
        self.c = c
        self.gamma = gamma
        self.constant = constant
        self.degree = degree

    def _extend_matrix(self):

        row = np.tile(self.k, (1, self.data_matrix.shape[1]))

        self.extended_data = np.vstack((self.data_matrix, row))

        return self.extended_data

    @staticmethod
    def _polynomial(data_1, data_2, constant, degree, k):

        result = (np.dot(data_1.T, data_2) + constant) ** degree + k ** 2

        return result

    @staticmethod
    def _rbf(data_1, data_2, gamma, k):

        result = np.exp(-gamma * (np.linalg.norm(data_1 - data_2) ** 2)) + k ** 2

        return result

    def _calculate_h(self):

        self.z = 2 * self.labels - 1

        self.Extended_Data = self._extend_matrix()

        if self.mode == "Linear":
            g = np.dot(self.extended_data.T, self.extended_data)

        elif self.mode == "Kernel Polynomial":
            g = self._polynomial(self.data_matrix, self.data_matrix, self.constant, self.degree, self.k)

        else:
            g = np.zeros((self.data_matrix.shape[1], self.data_matrix.shape[1]))
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    g[i, j] = self._rbf(self.data_matrix[:, i], self.data_matrix[:, j], self.gamma, self.k)

        h = np.dot(self.z.reshape(self.z.shape[0], 1), self.z.reshape(self.z.shape[0], 1).T) * g

        return h

    def _l_function(self, alpha):

        h = self._calculate_h()
        ones = np.ones(self.extended_data.shape[1])

        result = 0.5 * np.dot(np.dot(alpha.T, h), alpha) - np.dot(alpha.T, ones)

        gradient = np.dot(h, alpha) - ones

        return result, gradient.reshape(gradient.size)

    def _find_alpha(self):

        self.extended_data = self._extend_matrix()

        bound = [(0, self.c)] * self.Extended_Data.shape[1]
        x0 = np.zeros(self.Extended_Data.shape[1])

        x, self.Primal_loss, _ = scipy.optimize.fmin_l_bfgs_b(self._l_function, x0=x0, bounds=bound, factr=1.0)

        return x, self.primal_loss

    def predict(self, test_matrix):

        score = np.zeros(test_matrix.shape[1])

        alpha, self.primal_loss = self._find_alpha()

        if self.mode == "Linear":

            self.w_hat = np.dot(self.z * self.extended_data, alpha)
            w = self.w_hat[:test_matrix.shape[0]].reshape(self.w_hat[:test_matrix.shape[0]].shape[0], 1)
            b = self.w_hat[DTE.shape[0]]

            for i in range(DTE.shape[1]):
                score[i] = np.dot(w.T, test_matrix[:, i]) + b

        elif self.mode == "Kernel Polynomial":
            score = np.dot(alpha * self.z, self._polynomial(self.data_matrix, test_matrix, self.constant, self.degree,
                                                            self.k))
        elif self.mode == "Kernel RBF":

            kernel_values = np.zeros((self.data_matrix.shape[1], test_matrix.shape[1]))

            for i in range(self.data_matrix.shape[1]):
                for j in range(DTE.shape[1]):
                    kernel_values[i, j] = self._rbf(self.data_matrix[:, i], test_matrix[:, j], self.gamma, self.k)

            score = np.dot(alpha * self.z, kernel_values)

        self.predicted_labels = np.int32(score > 0)

        return self.predicted_labels

    def _loss_for_linear(self):

        maxsum = 0

        for i in range(self.Extended_Data.shape[1]):
            maxsum += max(0, 1 - (self.z[i] * np.dot(self.w_hat, self.extended_data[:, i])))

        self.dual_loss = 0.5 * (np.linalg.norm(self.w_hat)) ** 2 + self.c * maxsum

        dual_gap = self.dual_loss + self.Primal_loss

        return self.dual_loss, self.Primal_loss, dual_gap

    def _loss_for_kernel(self):

        self.dual_loss = -self.primal_loss

        return self.dual_loss

    def calculate_losses(self):

        if self.mode == "Linear":
            return self._loss_for_linear()

        if self.mode == "Kernel Polynomial":
            return self._loss_for_kernel()

        if self.mode == "Kernel RBF":
            return self._loss_for_kernel()

    def calculate_error(self, test_labels):

        bool_predictions = np.array(self.predicted_labels != test_labels)

        error = float(bool_predictions.sum() / test_labels.shape[0])

        return error * 100

    def calculate_accuracy(self, test_labels):

        bool_predictions = np.array(self.predicted_labels == test_labels)

        accuracy = float(bool_predictions.sum() / test_labels.shape[0])

        return accuracy * 100


if __name__ == "__main__":
    Data, Label = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(Data, Label)

    K_array = [1, 10]
    C_array = [0.1, 1, 10]

    K_array_Kernel = [0, 1]
    constant_value = [0, 1]
    gamma_value = [1, 10]

    # ----------------Linear SVM----------------#

    """print("K | C | Primal Loss | Dual Loss | Duality Gap | Error Rate\n ")

    for K in K_array:
        for C in C_array:
            svm = SVM("Linear")
            svm.fit(DTR, LTR, K, C)
            Predictions = svm.Predict(DTE)
            Dual_loss,Primal_loss,Duality_gap = svm.CalculateLosses()
            error_rate = svm.calculate_error(LTE)

            print(f"{K} | {C} | {-Primal_loss:.6e} | {Dual_loss:.6e} | {Duality_gap:6e} | {error_rate:.1f}%\n ")"""

    # --------------Polynomial Kernel SVM --------#

    """print("K | C | Kernel: Polynomial | Dual Loss | Error Rate\n ")

    for k in K_array_Kernel:
        for constant in constant_value:
            svm = SVM("Kernel Polynomial")
            svm.fit(DTR,LTR,k,1,constant= constant,degree = 2)
            Predictions = svm.Predict(DTE)
            Dual_loss = svm.CalculateLosses()
            error_rate = svm.calculate_error(LTE)

            print(f"{k} | {1} | (d={2},c={constant}) | {Dual_loss:.6e} | {error_rate:.1f}%\n ")"""

    # -------------------RBF Kernel SVM------------------------#

    """print("K | C | Kernel: RBF | Dual Loss | Error Rate\n ")

    for k in K_array_Kernel:
        for gamma in gamma_value:
            svm = SVM("Kernel RBF")
            svm.fit(DTR,LTR,k,1,gamma = gamma)
            Predictions = svm.Predict(DTE)
            Dual_loss = svm.CalculateLosses()
            error_rate = svm.calculate_error(LTE)

            print(f"{k} | {1} | (gamma={gamma}) | {Dual_loss:.6e} | {error_rate:.1f}%\n ")"""
