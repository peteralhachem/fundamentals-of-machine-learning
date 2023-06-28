import numpy as np
import matplotlib.pyplot as plt
from Utils import Load_Data_from_file,Calculate_mean,Center_Data,scatter_plot,Calculate_Covarariance_Matrix




class PCA: 
    def __init__(self, n_components):
        self.n_components = None
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self,X): 
        #---Center the data and calculate the mean---#
        centered_data = Center_Data(X)
        self.mean = Calculate_mean(X)
    

        #Compute the covariance matrix
        CovarianceMatrix = Calculate_Covarariance_Matrix(centered_data)

        #Compute the eigenvalues and eigenvectors
        eigenvalues,eigenvectors = np.linalg.eigh(CovarianceMatrix)

        self.components = eigenvectors[:, ::-1][:, :self.n_components]


    def transform(self,X):
        #---Center the data and calculate the mean---#
        centered_data =Center_Data(X)


        X_transformed = np.dot(self.components.T,centered_data)

        return X_transformed







    
    



   



if __name__ == '__main__':

    data,labels = Load_Data_from_file("Dataset/iris.csv")

    pca = PCA(n_components=2)
    pca.fit(X=data)
    X_transformed = pca.transform(X=data)

    scatter_plot(X_transformed,labels)















