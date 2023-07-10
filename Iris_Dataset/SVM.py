import numpy as np
from Utils import *
import scipy
import matplotlib.pyplot as plt


class SVM:
    def __init__(self,mode,Data,Label,K,C,gamma= None,constant= None,degree=None):
        self.mode = mode
        self.Data = Data
        self.Label = Label
        self.K = K
        self.C = C
        self.gamma = gamma
        self.constant = constant
        self.degree = degree
        self.Z = None
        self.Extended_Data = None
        self.Predicted_Label = None
        self.Primal_loss = None
        self.Dual_loss = None
        self.W_hat = None




    def _ExtendMatrix(self):

        row = np.tile(self.K,(1,self.Data.shape[1]))

        self.Extended_Data = np.vstack((self.Data,row))

        return self.Extended_Data

    def _Polynomial_Kernel(self,Data_1, Data_2, constant, degree, K):

        result = (np.dot(Data_1.T, Data_2) + constant) ** degree + K ** 2

        return result

    def _RBF_Kernel(self,Data_1, Data_2, gamma, K):
        result = np.exp(-gamma * (np.linalg.norm(Data_1 - Data_2) ** 2)) + K ** 2

        return result

    def _Calculate_H(self):

        self.Z = 2 * self.Label - 1

        self.Extended_Data = self._ExtendMatrix()

        if self.mode == "Linear":
            G = np.dot(self.Extended_Data.T,self.Extended_Data)

        elif self.mode == "Kernel Polynomial":
            G = self._Polynomial_Kernel(self.Data,self.Data,self.constant,self.degree,self.K)

        elif self.mode == "Kernel RBF":
            G = np.zeros((self.Data.shape[1],self.Data.shape[1]))
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    G[i,j] = self._RBF_Kernel(self.Data[:,i],self.Data[:,j],self.gamma,self.K)


        H = np.dot(self.Z.reshape(self.Z.shape[0],1),self.Z.reshape(self.Z.shape[0],1).T) * G

        return H

    def _L(self,Alpha):

        H = self._Calculate_H()
        ones = np.ones(self.Extended_Data.shape[1])
        result = 0.5 * np.dot(np.dot(Alpha.T,H), Alpha) - np.dot(Alpha.T,ones)

        gradient = np.dot(H,Alpha) - ones


        return result, gradient.reshape(gradient.size)

    def _Find_Alpha(self):

        self.Extended_Data = self._ExtendMatrix()
        bound = [(0, self.C)] * self.Extended_Data.shape[1]
        x0 = np.zeros(self.Extended_Data.shape[1])

        X, self.Primal_loss, _ = scipy.optimize.fmin_l_bfgs_b(self._L, x0=x0, bounds=bound, factr=1.0)

        return X,self.Primal_loss

    def Predict(self,DTE):

        Score = np.zeros(DTE.shape[1])

        Alpha,self.Primal_loss = self._Find_Alpha()

        if self.mode == "Linear":

            self.W_hat = np.dot(self.Z * self.Extended_Data, Alpha)
            W = self.W_hat[:DTE.shape[0]].reshape(self.W_hat[:DTE.shape[0]].shape[0], 1)
            b = self.W_hat[DTE.shape[0]]

            for i in range(DTE.shape[1]):
                Score[i] = np.dot(W.T, DTE[:, i]) + b

        elif self.mode == "Kernel Polynomial":
            Score = np.dot(Alpha * self.Z, self._Polynomial_Kernel(self.Data,DTE,self.constant,self.degree,self.K))

        elif self.mode == "Kernel RBF":
            Kernel_values = np.zeros((self.Data.shape[1],DTE.shape[1]))

            for i in range(self.Data.shape[1]):
                for j in range(DTE.shape[1]):
                    Kernel_values[i,j] = self._RBF_Kernel(self.Data[:,i],DTE[:,j],self.gamma,self.K)


            Score = np.dot(Alpha * self.Z, Kernel_values)



        self.Predicted_Label = np.int32(Score > 0)



        return self.Predicted_Label



    def _CalculateLossesForLinear(self):

        Max_Sum = 0

        for i in range(self.Extended_Data.shape[1]):
            Max_Sum += max(0, 1 - (self.Z[i] * np.dot(self.W_hat, self.Extended_Data[:, i])))

        self.Dual_loss = 0.5 * (np.linalg.norm(self.W_hat)) ** 2 + self.C * Max_Sum

        Dual_Gap = self.Dual_loss + self.Primal_loss

        return self.Dual_loss, self.Primal_loss, Dual_Gap

    def _CalculateLossesforkernel(self):

        self.Dual_loss = -self.Primal_loss

        return self.Dual_loss

    def CalculateLosses(self):

        if self.mode == "Linear":
            return self._CalculateLossesForLinear()

        if self.mode == "Kernel Polynomial":
            return self._CalculateLossesforkernel()

        if self.mode == "Kernel RBF":
            return self._CalculateLossesforkernel()




    def CalculateError(self,LTE):

        Bool_Predictions = np.array(self.Predicted_Label != LTE)

        error = float(Bool_Predictions.sum() / LTE.shape[0])

        return error * 100

    def CalculateAccuracy(self,LTE):

        Bool_Predictions = np.array(self.Predicted_Label == LTE)

        accuracy = float(Bool_Predictions.sum() / LTE.shape[0])

        return accuracy * 100


def Polynomial_Kernel(Data_1, Data_2, constant, degree, K):

    result = (np.dot(Data_1.T, Data_2) + constant) ** degree + K ** 2

    return result

def RBF_Kernel(Data_1, Data_2, gamma, K):
    result = np.exp(-gamma * (np.linalg.norm(Data_1 - Data_2) ** 2)) + K ** 2

    return result










if __name__ == "__main__":

    Data, Label = load_iris_binary()
    (DTR, LTR) , (DTE, LTE) = split_db_2to1(Data,Label)

    K_array =[1,10]
    C_array = [0.1,1,10]

    K_array_Kernel = [0,1]
    constant_value = [0,1]
    gamma_value = [1,10]

    #----------------Linear SVM----------------#

    """print("K | C | Primal Loss | Dual Loss | Duality Gap | Error Rate\n ")

    for K in K_array:
        for C in C_array:
            svm = SVM("Linear",DTR, LTR, K, C)
            Predictions = svm.Predict(DTE)
            Dual_loss,Primal_loss,Duality_gap = svm.CalculateLosses()
            error_rate = svm.CalculateError(LTE)

            print(f"{K} | {C} | {-Primal_loss:.6e} | {Dual_loss:.6e} | {Duality_gap:6e} | {error_rate:.1f}%\n ")"""



    #--------------Polynomial Kernel SVM --------#

    """print("K | C | Kernel: Polynomial | Dual Loss | Error Rate\n ")

    for k in K_array_Kernel:
        for constant in constant_value:
            svm = SVM("Kernel Polynomial",DTR,LTR,k,1,constant= constant,degree = 2)
            Predictions = svm.Predict(DTE)
            Dual_loss = svm.CalculateLosses()
            error_rate = svm.CalculateError(LTE)

            print(f"{k} | {1} | (d={2},c={constant}) | {Dual_loss:.6e} | {error_rate:.1f}%\n ")"""



    #-------------------RBF Kernel SVM------------------------#

    """print("K | C | Kernel: RBF | Dual Loss | Error Rate\n ")

    for k in K_array_Kernel:
        for gamma in gamma_value:
            svm = SVM("Kernel RBF",DTR,LTR,k,1,gamma = gamma)
            Predictions = svm.Predict(DTE)
            Dual_loss = svm.CalculateLosses()
            error_rate = svm.CalculateError(LTE)

            print(f"{k} | {1} | (gamma={gamma}) | {Dual_loss:.6e} | {error_rate:.1f}%\n ")"""



            




















