import numpy as np
from sklearn.datasets import load_iris
from Utils import *



class Gaussian_Classifier():
    def __init__(self,mode):
        self.mode = mode
        self.data = None
        self.label = None
        self.mean = None
        self.covariance = None
        self.Predicted_Labels = None

    def fit(self,data,label):
        self.data = data
        self.label = label
        self.mean = Calculate_class_means(self.data,self.label)
        self.covariance = 0

        if self.mode == "Tied":
            for value in np.unique(self.label):
                inter = Calculate_Covarariance_Matrix(Center_Data(self.data[:, self.label == value]))
                self.covariance += (inter * (self.data[:, self.label == value].shape[1]))

            self.covariance = self.covariance / data.shape[1]

        elif self.mode == "Multivariate":

            self.covariance = Calculate_class_covariances(self.data,self.label)

        elif self.mode == "Naive Bayes":

            self.covariance = Calculate_class_covariances(self.data, self.label)
            self.covariance = self.covariance * np.identity(self.data.shape[0])

        elif self.mode == "Tied Naive Bayes":
            for value in np.unique(self.label):
                inter = Calculate_Covarariance_Matrix(Center_Data(self.data[:, self.label == value]))
                self.covariance += (inter * (self.data[:, self.label == value].shape[1]))

            self.covariance = self.covariance / data.shape[1]
            self.covariance = self.covariance * np.identity(self.data.shape[0])





    def predict(self,X):
        Prior_probability = np.array([1/3,1/3,1/3])
        likelihood_Scores = Calculate_class_likelihood(X,self.mean,self.covariance,self.mode)


        Joint_Densities = likelihood_Scores * Prior_probability.reshape(Prior_probability.size,1)

        Marginal_Densities = Joint_Densities.sum(0)
        Marginal_Densities = Marginal_Densities.reshape(1,Marginal_Densities.size)

        Posterior_Probability = Joint_Densities/Marginal_Densities

        self.Predicted_Labels = Posterior_Probability.argmax(axis=0)

        return self.Predicted_Labels


    def calculate_error(self,Y):

        Bool_Predictions = np.array(self.Predicted_Labels != Y)

        error = float(Bool_Predictions.sum()/Y.shape[0])

        return error*100

    def calculate_accuracy(self,Y):
        Bool_Predictions = np.array(self.Predicted_Labels == Y)

        accuracy = float(Bool_Predictions.sum() / Y.shape[0])

        return accuracy * 100


if __name__ == '__main__':
    Data,Label = Load_Iris()

    (DTR,LTR),(DTE,LTE) = split_db_2to1(Data,Label)

    models = ["Multivariate","Naive Bayes","Tied","Tied Naive Bayes"]

    for model in models:
        print("The error rate of the prediction of model " + model + " : {:.1f} %".format(LOO_Cross_Validation(Gaussian_Classifier,model,Data,Label)))
        print("\n---------------------------------\n")






