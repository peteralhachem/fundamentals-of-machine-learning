import numpy as np
from Utils import *
import scipy
import matplotlib.pyplot as plt

class MulticlassEvaluation():
    def __init__(self,class_log_likelihood,ground_truth,CostMatrix,PriorVector):
        self.class_log_likelihood = class_log_likelihood
        self.ground_truth = ground_truth
        self.CostMatrix = CostMatrix
        self.PriorVector = PriorVector
        self.Posterior_Probabilities = None
        self.MatrixofCosts = None


    def Predict(self):



        Joint_log_likelihoods = self.class_log_likelihood + np.log(self.PriorVector.reshape(self.PriorVector.size,1))

        Marginal_log_likelihoods = scipy.special.logsumexp(Joint_log_likelihoods, axis=0)

        self.Posterior_Probabilities = np.exp(Joint_log_likelihoods - Marginal_log_likelihoods)

        self.MatrixofCosts = np.dot(self.CostMatrix,self.Posterior_Probabilities)

        self.Predictions = np.argmin(self.MatrixofCosts, axis=0)

        return self.Predictions

    def CalculateDCF(self):

        cost_of_Dummy = np.min(np.dot(self.CostMatrix,self.PriorVector.reshape(self.PriorVector.size,1)))

        ConfusionMatrix = ComputeConfusionMatrix(self.Predictions,self.ground_truth)

        MissClassificationRatio = ConfusionMatrix/np.sum(ConfusionMatrix, axis = 0 )

        DCFu = np.zeros(self.PriorVector.size)

        for j in range(self.PriorVector.shape[0]):
            DCFu[j] = np.dot(np.dot(MissClassificationRatio[:, j] , self.CostMatrix[:, j]),self.PriorVector[j])




        return DCFu.sum(), DCFu.sum()/cost_of_Dummy









if __name__ == '__main__':

    ground_truth = np.load("Dataset/commedia_labels.npy")
    class_log_likelihoods = np.load("Dataset/commedia_ll.npy")

    class_log_likelihoods_eps1 = np.load("Dataset/commedia_ll_eps1.npy")
    ground_truth_eps1 = np.load("Dataset/commedia_labels_eps1.npy")

    Cost = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    Priors = np.array([0.3, 0.4, 0.3])

    multiclass = MulticlassEvaluation(class_log_likelihoods,ground_truth,Cost,Priors)

    Predicted_labels = multiclass.Predict()

    DCFu,DCF = multiclass.CalculateDCF()

    print("e=0.001")
    print("DCFu | DCF")
    print(f"{DCFu:.3f} | {DCF:.3f}\n")

    multiclass = MulticlassEvaluation(class_log_likelihoods_eps1, ground_truth_eps1, Cost, Priors)

    Predicted_labels = multiclass.Predict()

    DCFu, DCF = multiclass.CalculateDCF()

    print("e=1")
    print("DCFu | DCF")
    print(f"{DCFu:.3f} | {DCF:.3f}\n")

    print("============================================================\n")


    Cost = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    Priors = np.array([1/3, 1/3, 1/3])

    multiclass = MulticlassEvaluation(class_log_likelihoods, ground_truth, Cost, Priors)

    Predicted_labels = multiclass.Predict()

    DCFu, DCF = multiclass.CalculateDCF()

    print("e=0.001")
    print("DCFu | DCF")
    print(f"{DCFu:.3f} | {DCF:.3f}\n")

    multiclass = MulticlassEvaluation(class_log_likelihoods_eps1, ground_truth_eps1, Cost, Priors)

    Predicted_labels = multiclass.Predict()

    DCFu, DCF = multiclass.CalculateDCF()

    print("e=1")
    print("DCFu | DCF")
    print(f"{DCFu:.3f} | {DCF:.3f}\n")






