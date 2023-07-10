import numpy as np
from Utils import *
import scipy
import matplotlib.pyplot as plt



class BinaryTask:

    def __init__(self,log_likelihood_ratio):
        self.Log_Likelihood_ratio = log_likelihood_ratio
        self.Predictions = None
        self.ConfusionMatrix = None
        self.Prior_Class_Probability = None
        self.Cost_Matrix = None
        self.Ground_Truth = None

    def Predict(self,Cost_matrix,Prior_Class_Probability):

        self.Cost_Matrix = Cost_matrix
        self.Prior_Class_Probability = Prior_Class_Probability

        threshold = - np.log((self.Prior_Class_Probability*self.Cost_Matrix[0,1])/((1-self.Prior_Class_Probability)*self.Cost_Matrix[1,0]))

        self.Predictions = np.int32(self.Log_Likelihood_ratio > threshold)



        return self.Predictions

    def CalculateConfusionMatrix(self,ground_truth):

        self.Ground_Truth = ground_truth
        self.ConfusionMatrix = ComputeConfusionMatrix(self.Predictions,self.Ground_Truth)

        return self.ConfusionMatrix

    def ComputeBayesRisk(self):

        False_Negative_Rate = self.ConfusionMatrix[0,1]/(self.ConfusionMatrix[0,1]+self.ConfusionMatrix[1,1])
        False_Positive_Rate = self.ConfusionMatrix[1,0]/(self.ConfusionMatrix[1,0]+self.ConfusionMatrix[0,0])

        BayesRisk = (self.Prior_Class_Probability * self.Cost_Matrix[0,1] * False_Negative_Rate) + ((1-self.Prior_Class_Probability) * self.Cost_Matrix[1,0] * False_Positive_Rate)

        Normalized_BayesRisk = BayesRisk/min(self.Prior_Class_Probability * self.Cost_Matrix[0,1],(1-self.Prior_Class_Probability) * self.Cost_Matrix[1,0])

        return BayesRisk,Normalized_BayesRisk

    def Minimum_DCF(self,ground_truth):
        self.Ground_Truth = ground_truth
        thresholds = np.array(self.Log_Likelihood_ratio)
        thresholds.sort()
        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
        False_Negative_Rate = np.zeros((thresholds.size))
        False_Positive_Rate = np.zeros((thresholds.size))
        BayesRisk = np.zeros((thresholds.size))

        for index,threshold in enumerate(thresholds):

            self.Predictions = np.int32(self.Log_Likelihood_ratio > threshold)

            self.ConfusionMatrix = ComputeConfusionMatrix(self.Predictions,self.Ground_Truth)

            False_Negative_Rate[index] = self.ConfusionMatrix[0, 1] / (self.ConfusionMatrix[0, 1] + self.ConfusionMatrix[1, 1])
            False_Positive_Rate[index] = self.ConfusionMatrix[1, 0] / (self.ConfusionMatrix[1, 0] + self.ConfusionMatrix[0, 0])
            BayesRisk[index] = (self.Prior_Class_Probability * self.Cost_Matrix[0,1] * False_Negative_Rate[index]) + ((1-self.Prior_Class_Probability) * self.Cost_Matrix[1,0] * False_Positive_Rate[index])



        return min(BayesRisk)/min(self.Prior_Class_Probability * self.Cost_Matrix[0,1],(1-self.Prior_Class_Probability) * self.Cost_Matrix[1,0])


    def ROC_Curve(self):

        thresholds = np.array(self.Log_Likelihood_ratio)
        thresholds.sort()
        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
        False_Negative_Rate = np.zeros((thresholds.size))
        False_Positive_Rate = np.zeros((thresholds.size))

        for index,threshold in enumerate(thresholds):

            self.Predictions = np.int32(self.Log_Likelihood_ratio > threshold)

            self.ConfusionMatrix = ComputeConfusionMatrix(self.Predictions,self.Ground_Truth)

            False_Negative_Rate[index] = self.ConfusionMatrix[0, 1] / (self.ConfusionMatrix[0, 1] + self.ConfusionMatrix[1, 1])
            False_Positive_Rate[index] = self.ConfusionMatrix[1, 0] / (self.ConfusionMatrix[1, 0] + self.ConfusionMatrix[0, 0])

        True_Positive_Rate = 1 - False_Negative_Rate

        plt.plot(False_Positive_Rate, True_Positive_Rate)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.show()




def UselessPrediction(class_log_likelihoods):
    # ---Maxiumum posterior probabilities---#

    Joint_log_likelihoods = class_log_likelihoods + np.log(1.0 / 3.0)

    Marginal_log_likelihoods = scipy.special.logsumexp(Joint_log_likelihoods, axis=0)

    Posterior_probabilities = np.exp(Joint_log_likelihoods - Marginal_log_likelihoods)

    Predictions = np.argmax(Posterior_probabilities, axis=0)

    return Predictions


if __name__ == '__main__':

    ground_truth = np.load("Dataset/commedia_labels.npy")
    class_log_likelihoods = np.load("Dataset/commedia_ll.npy")

    Log_Likelihood_ratio = np.load("Dataset/commedia_llr_infpar.npy")
    binary_label = np.load("Dataset/commedia_labels_infpar.npy")

    Log_Likelihood_ratio_eps1 = np.load("Dataset/commedia_llr_infpar_eps1.npy")
    binary_label_eps1 = np.load("Dataset/commedia_labels_infpar_eps1.npy")

    Prior_Class_Probabilities = [0.5,0.8,0.5,0.8]

    Cost_Matrix = np.array([[[0,1],[1,0]],[[0,1],[1,0]],[[0,10],[1,0]],[[0,1],[10,0]]])


    for i in range(len(Prior_Class_Probabilities)):

        binary_task = BinaryTask(Log_Likelihood_ratio)
        Predicted_Labels = binary_task.Predict(Cost_Matrix[i],Prior_Class_Probabilities[i])
        Confusion_Matrix = binary_task.CalculateConfusionMatrix(binary_label)
        BayesRisk, NormalizedBayesRisk = binary_task.ComputeBayesRisk()
        Min_DCF = binary_task.Minimum_DCF(binary_label)


        print("e = 0.001")
        print("(Cfn,Cfp,Pi) | BayesRisk | Normalized BayesRisk | Minimun_DCF ")
        print(f"({Cost_Matrix[i][0,1]},{Cost_Matrix[i][1,0]},{Prior_Class_Probabilities[i]}) | {BayesRisk:.3f} | {NormalizedBayesRisk:.3f} | {Min_DCF:.3f} \n")
        print("================================\n")
        print("Confusion Matrix\n")
        print(f"{Confusion_Matrix}\n")
        print("--------------------------------")
        
        binary_task = BinaryTask(Log_Likelihood_ratio_eps1)
        Predicted_Labels = binary_task.Predict(Cost_Matrix[i], Prior_Class_Probabilities[i])
        Confusion_Matrix = binary_task.CalculateConfusionMatrix(binary_label_eps1)
        BayesRisk, NormalizedBayesRisk = binary_task.ComputeBayesRisk()
        Min_DCF = binary_task.Minimum_DCF(binary_label_eps1)


        print("e = 1")
        print("(Cfn,Cfp,Pi) | BayesRisk | Normalized BayesRisk | Minimun_DCF ")
        print(
            f"({Cost_Matrix[i][0, 1]},{Cost_Matrix[i][1, 0]},{Prior_Class_Probabilities[i]}) | {BayesRisk:.3f} | {NormalizedBayesRisk:.3f} | {Min_DCF:.3f} \n")
        print("================================\n")
        print("Confusion Matrix\n")
        print(f"{Confusion_Matrix}\n")
        print("--------------------------------")


    effPriorLogOdds = np.linspace(-3, 3, 21)
    Effective_Prior = 1 / (1 + np.exp(-effPriorLogOdds))
    Cost_Matrix = np.array([[0, 1], [1, 0]])
    min_DCF = np.zeros((Effective_Prior.size))
    Normalized_DCF = np.zeros((Effective_Prior.size))
    min_DCF_1 = np.zeros((Effective_Prior.size))
    Normalized_DCF_1 = np.zeros((Effective_Prior.size))



    for index, Prior in enumerate(Effective_Prior):
        binary_task = BinaryTask(Log_Likelihood_ratio)
        Predictions = binary_task.Predict(Cost_Matrix, Prior)
        ConfusionMatrix = binary_task.CalculateConfusionMatrix(binary_label)
        _, Normalized_DCF[index] = binary_task.ComputeBayesRisk()
        min_DCF[index] = binary_task.Minimum_DCF(binary_label)

        binary_task = BinaryTask(Log_Likelihood_ratio_eps1)
        Predictions = binary_task.Predict(Cost_Matrix, Prior)
        ConfusionMatrix = binary_task.CalculateConfusionMatrix(binary_label_eps1)
        _, Normalized_DCF_1[index] = binary_task.ComputeBayesRisk()
        min_DCF_1[index] = binary_task.Minimum_DCF(binary_label_eps1)


    BayesErrorPlot(effPriorLogOdds,Normalized_DCF,min_DCF)
    BayesErrorPlot(effPriorLogOdds,Normalized_DCF,min_DCF,Normalized_DCF_1,min_DCF_1)












