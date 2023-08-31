from math import inf
from scipy.special import logsumexp

from Utils import *
from GMM_load import *


class GMM:

    def __init__(self, data_matrix):
        self.Data = data_matrix
        self.gmm_components = None
        self.responsibilities = None
        self.joint_log_density = None
        self.marginal_log_density = None

    def _gmm_log_density(self):

        self.joint_log_density = np.zeros((len(self.gmm_components), self.Data.shape[1]))

        for g in range(len(self.gmm_components)):
            self.joint_log_density[g, :] = MVG(self.Data, self.gmm_components[g][1], self.gmm_components[g][2])

            self.joint_log_density[g, :] += np.log(self.gmm_components[g][0])

        self.marginal_log_density = logsumexp(self.joint_log_density, axis=0)

        return self.marginal_log_density, self.joint_log_density

    def _e_step(self):

        self.marginal_log_density, self.joint_log_density = self._gmm_log_density()

        self.responsibilities = np.exp(self.joint_log_density - self.marginal_log_density)

        log_likelihood_value = np.sum(self.marginal_log_density)

        return self.responsibilities, log_likelihood_value

    def _m_step(self):

        zero_order = np.zeros((len(self.gmm_components)))
        first_order = np.zeros((len(self.gmm_components), self.Data.shape[0]))
        second_order = np.zeros((len(self.gmm_components), self.Data.shape[0], self.Data.shape[0]))

        mu = np.zeros((len(self.gmm_components), self.Data.shape[0]))
        covariance = np.zeros((len(self.gmm_components), self.Data.shape[0], self.Data.shape[0]))
        mean_products = np.zeros((len(self.gmm_components), self.Data.shape[0], self.Data.shape[0]))
        # weights = np.zeros((len(self.gmm_components)))

        new_gmm = []

        for g in range(len(self.gmm_components)):
            zero_order[g] = self.responsibilities[g].sum()
            first_order[g] = np.dot(self.responsibilities[g].reshape((1, self.Data.shape[1])), self.Data.T)

        for g in range(len(self.gmm_components)):
            temp_matrix = np.zeros((self.Data.shape[0], self.Data.shape[0]))
            for i in range(self.Data.shape[1]):
                temp_matrix += self.responsibilities[g, i] * np.dot(self.Data[:, i].reshape((self.Data.shape[0], 1)),
                                                                    self.Data[:, i].reshape((1, self.Data.shape[0])))

            second_order[g] = temp_matrix

            mu[g] = first_order[g] / zero_order[g]

        for g in range(len(self.gmm_components)):
            mean_products[g] = np.dot(mu[g, :].reshape((self.Data.shape[0], 1)),
                                      mu[g, :].reshape((1, self.Data.shape[0])))

            covariance[g] = (second_order[g] / zero_order[g]) - mean_products[g]

        weights = zero_order / zero_order.sum()

        for g in range(len(self.gmm_components)):
            new_gmm.append((weights[g], mu[g].reshape((self.Data.shape[0], 1)), covariance[g]))

        self.gmm_components = new_gmm

        return self.gmm_components

    def em_algorithm(self, gmm_component, delta_l=10 ** -6):

        self.gmm_components = gmm_component

        previous_log_likelihood = -inf
        num_samples = self.Data.shape[1]

        while True:
            self.responsibilities, current_log_likelihood = self._e_step()

            likelihood_increment = current_log_likelihood - previous_log_likelihood

            if likelihood_increment < delta_l * num_samples:
                average_log_likelihood = current_log_likelihood / num_samples
                return self.gmm_components, average_log_likelihood

            new_gmm_components = self._m_step()

            self.gmm_components = new_gmm_components

            previous_log_likelihood = current_log_likelihood

    def lbg_algorithm(self, num_components, alpha=0.1):
        """
        Performs the G-component decomposition into a 2G-component without the need to a point of initialization for the
        GMM components.
        :param alpha: value between the range [0,1] that helps computing the displacement vector d.
        :param num_components: number of GMM components to be resulted at the end of the LBG algorithm.
        TODO: compute the mean and covariance matrix of the dataset.
        TODO: compute the displacement vector d.
        TODO: split the G-components in 2G-components.
        TODO: Update the value of the GMM.
        """

        mu = calculate_mean(self.Data)
        cov = calculate_covariance(self.Data)
        iteration = 0

        # ---Compute displacement vector d--- #

        u, s, _ = numpy.linalg.svd(cov)
        d = u[:, 0:1] * s[0] ** 0.5 * alpha

        self.gmm_components = [(1.0, mu, cov)]

        while iteration < num_components:

            updated_gmm, log_likelihood_value = self.em_algorithm(self.gmm_components)








    def __str__(self):

        return f"The current GMM component is: {self.gmm_components}."


if __name__ == "__main__":
    data_4D = np.load("GMM_data_4D.npy")
    data_1D = np.load("GMM_data_1D.npy")

    gmm_4D = load_gmm("GMM_4D_3G_init.json")
    gmm_1D = load_gmm("GMM_1D_3G_init.json")

    log_densities_4D = np.load("GMM_4D_3G_init_ll.npy")
    log_densities_1D = np.load("GMM_1D_3G_init_ll.npy")

    gmm = GMM(data_4D)

    gmm_components, log_likelihood = gmm.em_algorithm(gmm_4D)
