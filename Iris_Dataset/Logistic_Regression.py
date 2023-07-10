import numpy as np
import scipy.optimize
from Utils import *

class LogisticRegression:
    def __init__(self,DTR,LTR,value_lambda):
        self.DTR = DTR
        self.LTR = LTR
        self.value_lambda = value_lambda
        self.W = None
        self.b = None
        self.J = None
        self.Predicted_Labels = None

    def LogisticRegressionObjectiveFunction(self,V):

        W = V[0:-1]
        b = V[-1]
        log_value = 0

        for i in range(self.DTR.shape[1]):
            zi = 2 * LTR[i] -1
            log_value += np.logaddexp(0,-zi*(np.dot(W.T,self.DTR[:,i])+b))

        value = log_value/self.DTR.shape[1]
        value_1 = 0.5 * self.value_lambda * (np.linalg.norm(W)**2)

        result = value + value_1

        return result

    def fit(self):

        x0 = np.zeros(self.DTR.shape[0]+1)

        logistic_regression = LogisticRegression(self.DTR,self.LTR,self.value_lambda)

        X, self.J, _ = scipy.optimize.fmin_l_bfgs_b(logistic_regression.LogisticRegressionObjectiveFunction, x0, approx_grad=True,
                                               factr=10000000.0)

        self.W = X[:DTR.shape[0]]
        self.b = X[-1]

        return self.J



    def Predict(self,DTE):

        Score = np.zeros(DTE.shape[1])
        self.Predicted_Labels = np.zeros(DTE.shape[1])

        for i in range(DTE.shape[1]):
            Score[i] = np.dot(self.W, DTE[:, i]) + self.b

        for i in range(DTE.shape[1]):
            if Score[i] > 0:
                self.Predicted_Labels[i] = 1
            else:
                self.Predicted_Labels[i] = 0

        return self.Predicted_Labels

    def Calculate_error(self,LTE):

        boolean_values = np.array([LTE != self.Predicted_Labels])

        error_rate = boolean_values.sum() / LTE.shape[0]

        return error_rate*100


if __name__ == '__main__':

    Data, Label = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(Data, Label)

    lambda_values = [10**-6,1e-3,1e-1,1]

    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        logistic_regression = LogisticRegression(DTR, LTR,value)
        J = logistic_regression.fit()
        Predicted_labels = logistic_regression.Predict(DTE)
        error_rate = logistic_regression.Calculate_error(LTE)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6f} | {error_rate:.1f}%\n")














