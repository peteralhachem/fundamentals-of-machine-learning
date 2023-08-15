import numpy as np
import scipy.optimize
from Utils import *

class LogisticRegression:
    def __init__(self, mode):
        self.mode = mode
        self.DTR = None
        self.LTR = None
        self.value_lambda = None
        self.W = None
        self.b = None
        self.J = None
        self.Predicted_Labels = None

    def _LogisticRegressionObjectiveFunction(self,V):

        W = V[0:-1]
        b = V[-1]
        log_value = 0

        for i in range(self.DTR.shape[1]):
            zi = 2 * LTR[i] -1
            log_value += np.logaddexp(0,-zi*(np.dot(W.T,self.DTR[:,i])+b))

        logistic_loss = log_value/self.DTR.shape[1]
        regularization_term = 0.5 * self.value_lambda * (np.linalg.norm(W)**2)

        empirical_risk = regularization_term + logistic_loss

        return empirical_risk

    def _MulticlassLogisticRegressionObjectivefunction(self,V):

        T = np.zeros((len(np.unique(self.LTR)),self.DTR.shape[1]))

        for i in range(self.LTR.shape[0]):
            for k in np.unique(LTR):
                if LTR[i] == k:
                    T[k, i] = 1

        dimension = len(np.unique(self.LTR))
        self.W = V[:-(dimension)]
        self.b = V[-(dimension):]

        self.W = self.W.reshape(DTR.shape[0],len(np.unique(self.LTR)))
        self.b = self.b.reshape(len(np.unique(self.LTR)),1)

        Score = np.dot(self.W.T,DTR) + self.b

        variable = np.log(np.sum(np.exp(Score),axis=0))

        log_Y = Score - variable

        value = T * log_Y

        final_value = np.sum(value)/DTR.shape[1]

        constant = (self.W*self.W).sum() * (0.5*self.value_lambda)

        result  = constant - final_value

        return result

    def fit(self,DTR,LTR,value_lambda):

        self.DTR = DTR
        self.LTR = LTR
        self.value_lambda = value_lambda

        if self.mode == "Binary":
            x0 = np.zeros(self.DTR.shape[0]+1)
            X, self.J, _ = scipy.optimize.fmin_l_bfgs_b(self._LogisticRegressionObjectiveFunction, x0, approx_grad=True,
                                               factr=10000000.0)
            self.W = X[:DTR.shape[0]]
            self.b = X[-1]

        elif self.mode == "Multiclass":
            x0 = np.zeros((self.DTR.shape[0] * len(np.unique(self.LTR))) + len(np.unique(self.LTR)))

            X, self.J, _ = scipy.optimize.fmin_l_bfgs_b(
                self._MulticlassLogisticRegressionObjectivefunction, x0,
                approx_grad=True,
                factr=10000000.0)

            self.W = X[:-len(np.unique(self.LTR))]
            self.b = X[-len(np.unique(self.LTR)):]

            self.W = self.W.reshape(DTR.shape[0], len(np.unique(self.LTR)))
            self.b = self.b.reshape(len(np.unique(self.LTR)), 1)


        return self.J



    def Predict(self,DTE):

        if self.mode == "Binary":
            Score = np.zeros(DTE.shape[1])
            self.Predicted_Labels = np.zeros(DTE.shape[1])

            for i in range(DTE.shape[1]):
                Score[i] = np.dot(self.W, DTE[:, i]) + self.b

            for i in range(DTE.shape[1]):
                if Score[i] > 0:
                    self.Predicted_Labels[i] = 1
                else:
                    self.Predicted_Labels[i] = 0

        elif self.mode == "Multiclass":
            Score = np.zeros((self.b.shape[0], DTE.shape[1]))
            self.Predicted_Labels = np.zeros(DTE.shape[1])

            for k in range(self.b.shape[0]):
                for i in range(DTE.shape[1]):
                    Score[k, i] = np.dot(self.W[:, k], DTE[:, i]) + self.b[k]

            self.Predicted_Labels = np.argmax(Score, axis = 0)

        return self.Predicted_Labels

    def Calculate_error(self,LTE):

        boolean_values = np.array([LTE != self.Predicted_Labels])

        error_rate = boolean_values.sum() / LTE.shape[0]

        return error_rate * 100


if __name__ == '__main__':

    Data_Binary, Label_Binary = load_iris_binary()
    (DTR_b, LTR_b), (DTE_b, LTE_b) = split_db_2to1(Data_Binary, Label_Binary)

    D, L = Load_Iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    lambda_values = [10**-6,1e-3,1e-1,1] #Used for regularization

    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        logistic_regression = LogisticRegression("Binary")
        J = logistic_regression.fit(DTR_b, LTR_b,value)
        Predicted_labels = logistic_regression.Predict(DTE_b)
        error_rate = logistic_regression.Calculate_error(LTE_b)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6e} | {error_rate:.1f}%\n")

    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        logistic_regression = LogisticRegression("Multiclass")
        J = logistic_regression.fit(DTR, LTR, value)
        Predicted_labels = logistic_regression.Predict(DTE)
        error_rate = logistic_regression.Calculate_error(LTE)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6e} | {error_rate:.1f}%\n")















