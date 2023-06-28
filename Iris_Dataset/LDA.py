
import numpy as np
from Utils import Load_Data_from_file,Calculate_mean,Center_Data,scatter_plot,Calculate_Covarariance_Matrix
import scipy.linalg


#---U can either use the generalized method of to get the eigenvalues and eigenvectors or you can use the  joint diagonalization---#

class LDA:
    def __init__(self,n_components):
        self.mean = None
        self.n_components = None
        self.n_components = n_components
        self.S_B = 0
        self.S_W = 0
        self.W = None


    def fit(self,X,labels):
        self.mean = Calculate_mean(X)

        for label in np.unique(labels):
           total_mean = Calculate_mean(X[:,labels == label]) - self.mean
           variable  =np.dot(total_mean,total_mean.T)*X[:,labels == label].shape[1]
           self.S_B += variable

           new_value = X[:,labels == label] - Calculate_mean(X[:,labels == label])
           S_W_c = Calculate_Covarariance_Matrix(new_value)

           self.S_W += (S_W_c*X[:,labels == label].shape[1])

        self.S_B = self.S_B/X.shape[1]
        self.S_W = self.S_W/X.shape[1]

        #Generalized eigenvalue problem
        #eigenvalues ,eigenvectors = scipy.linalg.eigh(self.S_B, self.S_W)

        #self.W = eigenvectors[:,::-1][:,0:self.n_components]

        #Joint Diagonalization of S_B and S_W
        #We use the single value divsion
        U,s,_ = scipy.linalg.svd(self.S_W)
        P1 = np.dot(np.dot(U,np.diag(1/(s**0.5))),U.T)
        S_B_T = np.dot(np.dot(P1,self.S_B),P1.T)
        s,P2 = scipy.linalg.eigh(S_B_T)

        self.W = U[:,::-1][:,0:self.n_components]





    def transform(self,X):
        X_centered = Center_Data(X)

        X_transformed = np.dot(self.W.T, X_centered)

        return X_transformed











if __name__ == '__main__':

    data,labels = Load_Data_from_file("Dataset/iris.csv")

    lda = LDA(n_components=2)
    lda.fit(data,labels)
    X_transformed = lda.transform(data)

    scatter_plot(X_transformed,labels)
