from scipy.optimize import fmin_l_bfgs_b
from Utils import *


class LogisticRegression:
    def __init__(self, mode):
        self.mode = mode
        self.data_matrix = None
        self.labels = None
        self.value_lambda = None
        self.W = None
        self.b = None
        self.J = None
        self.predicted_labels = None

    def _logistic_regression_objective_function(self, v):

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

        t_matrix = np.zeros((len(np.unique(self.labels)), self.data_matrix.shape[1]))

        for i in range(self.labels.shape[0]):
            for k in np.unique(self.labels):
                if self.labels[i] == k:
                    t_matrix[k, i] = 1

        dimension = len(np.unique(self.labels))
        self.W = v[:-dimension]
        self.b = v[-dimension:]

        self.W = self.W.reshape(DTR.shape[0], len(np.unique(self.labels)))
        self.b = self.b.reshape(len(np.unique(self.labels)), 1)

        score = np.dot(self.W.T, self.data_matrix) + self.b

        variable = np.log(np.sum(np.exp(score), axis=0))

        log_y = score - variable

        result = t_matrix * log_y

        final_value = np.sum(result) / self.data_matrix.shape[1]

        constant = (self.W * self.W).sum() * (0.5 * self.value_lambda)

        result = constant - final_value

        return result

    def fit(self, data_matrix, labels, value_lambda):

        self.data_matrix = data_matrix
        self.labels = labels
        self.value_lambda = value_lambda

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

        return self.J

    def predict(self, test_matrix):

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
                for i in range(DTE.shape[1]):
                    score[k, i] = np.dot(self.W[:, k], test_matrix[:, i]) + self.b[k]

            self.predicted_labels = np.argmax(score, axis=0)

        return self.predicted_labels

    def calculate_error(self, test_labels):

        boolean_values = np.array([test_labels != self.predicted_labels])

        error = boolean_values.sum() / LTE.shape[0]

        return error * 100


if __name__ == '__main__':

    data_binary, label_binary = load_iris_binary()
    (DTR_b, LTR_b), (DTE_b, LTE_b) = split_db_2to1(data_binary, label_binary)

    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    lambda_values = [10 ** -6, 1e-3, 1e-1, 1]  # Used for regularization

    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        logistic_regression = LogisticRegression("Binary")
        J = logistic_regression.fit(DTR_b, LTR_b, value)
        Predicted_labels = logistic_regression.predict(DTE_b)
        error_rate = logistic_regression.calculate_error(LTE_b)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6e} | {error_rate:.1f}%\n")

    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        logistic_regression = LogisticRegression("Multiclass")
        J = logistic_regression.fit(DTR, LTR, value)
        Predicted_labels = logistic_regression.predict(DTE)
        error_rate = logistic_regression.calculate_error(LTE)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6e} | {error_rate:.1f}%\n")
