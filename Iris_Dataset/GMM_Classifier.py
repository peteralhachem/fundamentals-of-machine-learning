import numpy as np
from Utils import *
from GMM import *


class GMM_Classifier:

    def __init__(self, numberofcomponents, model = None,alpha = 0.1, psi = 0.01):

        self.Data = None
        self.Label = None
        self.numberofcomponents = numberofcomponents
        self.alpha = alpha
        self.model = model
        self.psi = psi



    def fit(self,Data,Label):

        self.Data = Data
        self.Label = Label

        self.all_classes_of_estimated_gmm = []

        self.all_classes_of_estimated_likelihood = []


        for label in np.unique(Label):

            class_sample = self.Data[:, self.Label == label]

            gmm = GMM_Evaluator()

            gmm_values, likelihood = gmm.LBG(self.numberofcomponents,class_sample,self.alpha,self.model,self.psi)

            self.all_classes_of_estimated_gmm.append(gmm_values)

            self.all_classes_of_estimated_likelihood.append(likelihood)


        return self.all_classes_of_estimated_gmm, self.all_classes_of_estimated_likelihood




if __name__ == "__main__":

    Data, Label = Load_Iris()

    (DTR,LTR) , (DTE,LTE) = split_db_2to1(Data, Label)


    classifer = GMM_Classifier(numberofcomponents = 4, model = "Tied Covariance")

    value_1,value_2 = classifer.fit(DTR, LTR)



