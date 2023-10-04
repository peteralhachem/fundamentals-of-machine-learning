from math import inf
from scipy.special import logsumexp

from utils import *
from GMM_load import *
from mvg import multivariate_gaussian


class GMM:

    def __init__(self):
        self.Data = None
        self.gmm_components = None
        self.responsibilities = None
        self.joint_log_density = None
        self.marginal_log_density = None

    def _gmm_log_density(self):
        """
        Calculate the Joint and Marginal log densities.

        :return:
        """

        self.joint_log_density = np.zeros((len(self.gmm_components), self.Data.shape[1]))

        for g in range(len(self.gmm_components)):
            self.joint_log_density[g, :] = multivariate_gaussian(self.Data, self.gmm_components[g][1],
                                                                 self.gmm_components[g][2])

            self.joint_log_density[g, :] += np.log(self.gmm_components[g][0])

        self.marginal_log_density = logsumexp(self.joint_log_density, axis=0)

        return self.marginal_log_density, self.joint_log_density

    def _e_step(self):
        """
        With the E-step, we calculate the responsibilities and the value of the log likelihood used later to  update
        gmm components.
        :return: responsibilities (exponential form of the subtraction the joint log density and marginal log density),
        log likelihood value.

        """

        self.marginal_log_density, self.joint_log_density = self._gmm_log_density()

        self.responsibilities = np.exp(self.joint_log_density - self.marginal_log_density)

        log_likelihood_value = np.sum(self.marginal_log_density)

        return self.responsibilities, log_likelihood_value

    def _m_step(self, covariance_type, psi):

        """

        Performs the M step of the EM algorithm. In the M step, you update the weights, mean and covariances based on
        statistical values that are computed based on the responsibilities calculated in the E step.
        :param covariance_type: The type of the covariance can be either "full", "diagonal" or "tied".
        :psi: parameter used to constraint the eigenvalues of the covariance matrix, in this way we can bind the log
        likelihood value, so it does not degenerate very high values.
        :return: updated gmm components.

        """

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

        if covariance_type == 'diagonal':
            for g in range(len(self.gmm_components)):
                covariance[g] = covariance[g] * np.eye(covariance[g].shape[0])

        elif covariance_type == 'tied':
            temp_matrix = np.zeros((self.Data.shape[0], self.Data.shape[0]))
            for g in range(len(self.gmm_components)):
                temp_matrix += zero_order[g] * covariance[g]
            for g in range(len(self.gmm_components)):
                covariance[g] = temp_matrix / self.Data.shape[1]

        # ---Perform Eigenvalue check on all the covariances ---#
        for g in range(len(self.gmm_components)):
            covariance[g] = self._check_eigenvalues(covariance[g], psi=psi)

        weights = zero_order / zero_order.sum()

        for g in range(len(self.gmm_components)):
            new_gmm.append((weights[g], mu[g].reshape((self.Data.shape[0], 1)), covariance[g]))

        self.gmm_components = new_gmm

        return self.gmm_components

    def em_algorithm(self, gmm_component, psi, delta_l=10 ** -6, covariance_type="full"):
        """
        Performing the EM algorithm, that helps update the GMM components and calculate the average log likelihood.
        :param gmm_component: Initial gmm component used to compute the responsibilities.
        :param psi: threshold used for the variance of the covariance to not have degenerate likelihood values.
        :param delta_l: The threshold used to compare the difference between two consecutive likelihood values.
        :param covariance_type: Type of covariance used in the GMM component:[full, diagonal, tied].
        :return: Updated GMM components and average log likelihood.
        """

        self.gmm_components = gmm_component

        previous_log_likelihood = -inf
        num_samples = self.Data.shape[1]

        while True:
            self.responsibilities, current_log_likelihood = self._e_step()

            likelihood_increment = current_log_likelihood - previous_log_likelihood

            if likelihood_increment < delta_l * num_samples:
                average_log_likelihood = current_log_likelihood / num_samples
                return self.gmm_components, average_log_likelihood

            new_gmm_components = self._m_step(covariance_type, psi)

            self.gmm_components = new_gmm_components

            previous_log_likelihood = current_log_likelihood

    def lbg_algorithm(self, data_matrix, num_components, psi=0.01, alpha=0.1, covariance_type="full"):
        """
        Performs the G-component decomposition into a 2G-component without the need to a point of initialization for the
        GMM components.
        :param data_matrix: The data matrix to used to calculate the G-component decomposition.
        :param num_components: number of GMM components to be resulted at the end of the LBG algorithm.
        :param alpha: value between the range [0,1] that helps computing the displacement vector d.
        :param psi: value greater than zero, help us constrain the eigenvalues of the covariance matrix, so we don't get
        degenerate values in the log likelihood.
        :param covariance_type: The type of covariance we want to have, e.g. "full, diagonal, tied".

        """
        self.Data = data_matrix
        mu = calculate_mean(self.Data)
        cov = calculate_covariance(self.Data)

        if covariance_type == 'diagonal':
            cov = cov * np.eye(cov.shape[0])

        cov = self._check_eigenvalues(cov, psi)
        iteration = 0
        log_likelihood_value = 0

        self.gmm_components = [(1.0, mu, cov)]

        while True:
            if iteration < num_components / 2:
                d = self._compute_d(self.gmm_components, alpha=alpha)
                split_components = self._split_gmm_components(self.gmm_components, d)
                self.gmm_components = split_components
                new_components, log_likelihood_value = self.em_algorithm(self.gmm_components, psi=psi,
                                                                         covariance_type=covariance_type)
                self.gmm_components = new_components
                iteration += 1

            else:
                return self.gmm_components, log_likelihood_value

    @staticmethod
    def _check_eigenvalues(cov, psi):
        """
        Function that checks the eigenvalues of the covariance matrix.
        :param cov: covariance matrix of the GMM component.
        :param psi: threshold used to calculate the eigenvalues of the covariance matrix.
        :return: Updated covariance matrix of the GMM component.
        """

        u, s, _ = numpy.linalg.svd(cov)
        s[s < psi] = psi
        cov = numpy.dot(u, s.reshape(s.size, 1) * u.T)

        return cov

    @staticmethod
    def _split_gmm_components(gmm_components, d):
        """
        Function that splits the gmm_components from K-GMM components to 2K-GMM components.
        :param gmm_components: The gmm_components containing: (weights, mean, covariance).
        :param d: displacement value of the split gmm_components.
        :return: new split gmm_components.

        """

        new_gmm_components = []

        for index, gmm_component in enumerate(gmm_components):
            first_component = (gmm_component[0] / 2, gmm_component[1] + d[index], gmm_component[2])
            new_gmm_components.append(first_component)

            second_component = (gmm_component[0] / 2, gmm_component[1] - d[index], gmm_component[2])
            new_gmm_components.append(second_component)

        return new_gmm_components

    @staticmethod
    def _compute_d(gmm_components, alpha):
        """
        Compute the displacement used to split the GMM components.
        :param gmm_components: The gmm_components containing: (weights, mean, covariance).
        :param alpha: value used to compute the displacement.
        :return: array-like shape of the displacement for each distinct GMM component.


        """

        # ---Compute displacement vector d--- #

        d_array = []

        for g in range(len(gmm_components)):
            u, s, _ = numpy.linalg.svd(gmm_components[g][2])
            d = u[:, 0:1] * s[0] ** 0.5 * alpha
            d_array.append(d)

        return d_array
