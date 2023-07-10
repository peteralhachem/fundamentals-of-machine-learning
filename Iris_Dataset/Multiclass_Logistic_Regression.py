import numpy as np
import scipy.optimize
from Utils import *


class MulticlassLogisticRegression:
    def __init__(self,DTR,LTR,value_of_lambda):
        self.DTR = DTR
        self.LTR = LTR
        self.value_of_lambda = value_of_lambda
        self.W = None
        self.b = None
        self.J = None



    def MulticlassLogisticRegressionObjectivefunction(self,V):

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

        constant = (self.W*self.W).sum() * (0.5*self.value_of_lambda)

        result  = constant - final_value

        return result



    def fit(self):

        x0 = np.zeros((self.DTR.shape[0]*len(np.unique(self.LTR)))+len(np.unique(self.LTR)))

        multiclass_logistic_regression = MulticlassLogisticRegression(self.DTR,self.LTR,self.value_of_lambda)

        X, self.J, _ = scipy.optimize.fmin_l_bfgs_b(multiclass_logistic_regression.MulticlassLogisticRegressionObjectivefunction, x0,
                                                    approx_grad=True,
                                                    factr=10000000.0)

        self.W = X[:-len(np.unique(self.LTR))]
        self.b = X[-len(np.unique(self.LTR)):]

        self.W = self.W.reshape(DTR.shape[0], len(np.unique(self.LTR)))
        self.b = self.b.reshape(len(np.unique(self.LTR)), 1)


        return self.J

    def Predict(self,DTE):

        Score = np.zeros((self.b.shape[0],DTE.shape[1]))
        self.Predicted_Labels = np.zeros(DTE.shape[1])


        for k in range(self.b.shape[0]):
            for i in range(DTE.shape[1]):
                Score[k,i] = np.dot(self.W[:,k], DTE[:, i]) + self.b[k]


        self.Predicted_Labels = np.argmax(Score,axis=0)

        return self.Predicted_Labels

    def Calculate_error(self,LTE):

        boolean_values = np.array([LTE != self.Predicted_Labels])

        error_rate = boolean_values.sum() / LTE.shape[0]

        return error_rate*100










if __name__ == '__main__':

    D, L = Load_Iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    lambda_values = [10 ** -6, 1e-3, 1e-1, 1]



    print("-----------------------------")
    print("\t \t| J(W,b) | Error Rate\n")

    for value in lambda_values:
        multiclass_logistic_regression = MulticlassLogisticRegression(DTR, LTR, value)
        J = multiclass_logistic_regression.fit()
        Predicted_labels = multiclass_logistic_regression.Predict(DTE)
        error_rate = multiclass_logistic_regression.Calculate_error(LTE)

        print("-----------------------------")
        print(f"Lambda:{value} | {J:.6f} | {error_rate:.1f}%\n")























