import numpy as np
import scipy.special

from Utils import *
from MVG import MVG
from GMM_load import load_gmm
from MVG import MVG


class GMM_Evaluator:
    def __init__(self):
        self.X = None
        self.gmm = None
        self.Posterior_probability = None
        self.log_likelihood = None
        self.Weight_matrix = None
        self.Mean_matrix = None
        self.Covariance_matrix = None




    def _GMM_logdensity(self,X,gmm):


        self.Joint_log_density = np.zeros((len(gmm),X.shape[1]))

        for g in range(len(gmm)):
            self.Joint_log_density[g,:] = MVG(X,gmm[g][1],gmm[g][2])
            self.Joint_log_density[g,:] += np.log(gmm[g][0])


        self.Marginal_log_density = scipy.special.logsumexp(self.Joint_log_density,axis = 0)




        return self.Joint_log_density, self.Marginal_log_density


    def _Estep(self):

        Joint_log_density, Marginal_log_density = self._GMM_logdensity(self.X,self.gmm)

        self.Posterior_Probability = np.exp(Joint_log_density - Marginal_log_density)

        self.log_likelihood = np.sum(Marginal_log_density) / self.X.shape[1]

        return self.log_likelihood



    def _Mstep(self):

        new_gmm = []


        #---Zero Order Statistics---#
        Z_g = np.sum(self.Posterior_Probability, axis=1)

        new_weights = Z_g / np.sum(Z_g)

        F_g = np.dot(self.Posterior_Probability, self.X.T).T

        S_g = np.zeros((len(self.gmm), self.X.shape[0], self.X.shape[0]))

        for g in range(len(self.gmm)):
            intermediate_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
            for i in range(self.X.shape[1]):
                intermediate_matrix += self.Posterior_Probability[g, i] * np.dot(
                    self.X[:, i].reshape((self.X.shape[0], 1)), self.X[:, i].reshape((1, self.X.shape[0])))

            S_g[g] = intermediate_matrix

        new_mu = F_g / Z_g

        mu_product = np.zeros((len(self.gmm), self.X.shape[0], self.X.shape[0]))
        for g in range(len(self.gmm)):
            mu_product[g] = np.dot(new_mu[:, g].reshape((self.X.shape[0], 1)),
                                   new_mu[:, g].reshape((1, self.X.shape[0])))

        new_covariance = S_g / Z_g.reshape((Z_g.size, 1, 1)) - mu_product

        for g in range(len(self.gmm)):

            if self.model == "Diagonal Covariance":

                new_covariance[g] = new_covariance[g] * np.eye(new_covariance[g].shape[0])

            elif self.model == "Tied Covariance":

                new_covariance[g] = np.dot(Z_g,new_covariance[g])/self.X.shape[1]

            if self.psi != None:
                U, s, _ = np.linalg.svd(new_covariance[g])
                s[s < self.psi] = self.psi
                new_covariance[g] = np.dot(U, s.reshape((s.size, 1)) * U.T)


            new_gmm.append((new_weights[g],new_mu[:, g].reshape((new_mu.shape[0], 1)), new_covariance[g]))


        return new_gmm

    def EM(self, X, gmm, model = None,threshold = 10**(-6), psi = None):

        self.model = model
        self.X = X
        self.gmm = gmm
        self.psi = psi


        condition = True

        while(condition):

            average_log_likelihood_1 = self._Estep()
            self.gmm = self._Mstep()

            average_log_likelihood_2 = self._Estep()

            if ((average_log_likelihood_2 - average_log_likelihood_1) < 0):
                print("Loglikelihood NOT INCREASING")

            elif ((average_log_likelihood_2 - average_log_likelihood_1) < threshold ):
                condition = False


        return self.gmm, average_log_likelihood_2

    def _Compute_d(self,Covariance_matrix, alpha):

        U , s, Vh = np.linalg.svd(Covariance_matrix)

        d = U[:,0:1] * np.sqrt(s[0]) * alpha

        return d

    def LBG(self,number_of_components,X,alpha,model = None,psi = None):

        self.X = X
        self.psi = psi
        self.model = model
        self.psi = psi

        self.Mean_matrix = Calculate_mean(self.X)
        self.Covariance_matrix = Calculate_Covarariance_Matrix(self.X)
        self.Weight_matrix = 1.0

        self.gmm = [(self.Weight_matrix, self.Mean_matrix, self.Covariance_matrix)]

        new_gmm, loglikelihood = self.EM(self.X, self.gmm, self.model, self.psi)

        self.gmm = new_gmm


        gmm = []
        likelihood = []

        likelihood.append(loglikelihood)


        for iteration in range(int(number_of_components/2)):

            for g in range(len(self.gmm)):

                dg = self._Compute_d(self.gmm[g][2], alpha)

                gmm_1 = (self.gmm[g][0]/2, self.gmm[g][1] + dg, self.gmm[g][2])
                gmm.append(gmm_1)


                gmm_2 = (self.gmm[g][0]/2.0, self.gmm[g][1] - dg, self.gmm[g][2])
                gmm.append(gmm_2)

                


            new_gmm_1, log_likelihood_1 = self.EM(self.X,gmm)

            self.gmm = new_gmm_1
            gmm = []

            likelihood.append(log_likelihood_1)





        return self.gmm, likelihood


if __name__ == '__main__':

    Dataset_4D = np.load("GMM_data_4D.npy")
    gmm_4D = load_gmm("GMM_4D_3G_init.json")
    log_density_true_4D = np.load("GMM_4D_3G_init_ll.npy")
    GMM_4D_EM = load_gmm("GMM_4D_3G_EM.json")

    Dataset_1D = np.load("GMM_data_1D.npy")
    gmm_1D = load_gmm("GMM_1D_3G_init.json")
    log_density_true_1D = np.load("GMM_1D_3G_init_ll.npy")
    GMM_1D_EM = load_gmm("GMM_1D_3G_EM.json")

    GMM_1D_4G_EM_LBG = load_gmm("GMM_1D_4G_EM_LBG.json")

    """GMM_model_4D = GMM_Evaluator()
    GMM_EM_4, avg_log_likelihood_4 = GMM_model_4D.EM(Dataset_4D,gmm_4D)
    Joint, Marginal = GMM_model_4D.PlotHistogram()"""



    GMM_model_1D = GMM_Evaluator()
    gmm, loglikelihood = GMM_model_1D.EM(Dataset_1D,gmm_1D)



















