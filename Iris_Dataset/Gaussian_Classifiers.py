
from Utils import *


class GaussianClassifier:
    def __init__(self, mode):
        self.mode = mode
        self.data = None
        self.labels = None
        self.mean = None
        self.covariance = None
        self.predicted_labels = None

    def fit(self, data_matrix, labels):
        self.data = data_matrix
        self.labels = labels
        self.mean = calculate_class_means(self.data, self.labels)
        self.covariance = 0

        if self.mode == "Tied":
            for value in np.unique(self.labels):
                temp_value = calculate_covariance(center_data(self.data[:, self.labels == value]))
                self.covariance += (temp_value * (self.data[:, self.labels == value].shape[1]))

            self.covariance = self.covariance / self.data.shape[1]

        elif self.mode == "Multivariate":

            self.covariance = calculate_class_covariances(self.data, self.labels)

        elif self.mode == "Naive Bayes":

            self.covariance = calculate_class_covariances(self.data, self.labels)
            self.covariance = self.covariance * np.identity(self.data.shape[0])

        elif self.mode == "Tied Naive Bayes":

            for value in np.unique(self.labels):

                inter = calculate_covariance(center_data(self.data[:, self.labels == value]))
                self.covariance += (inter * (self.data[:, self.labels == value].shape[1]))

            self.covariance = self.covariance / self.data.shape[1]
            self.covariance = self.covariance * np.identity(self.data.shape[0])

    def predict(self, test_matrix):
        prior = np.array([1/3, 1/3, 1/3])
        likelihood_scores = calculate_class_likelihood(test_matrix, self.mean, self.covariance, self.mode)

        joint_densities = likelihood_scores * prior.reshape(prior.size, 1)

        # ----Marginal is the summation of all the Joint densities of a sample within all the classes---- #

        marginal_densities = joint_densities.sum(0)
        marginal_densities = marginal_densities.reshape(1, marginal_densities.size)

        posterior_probabilities = joint_densities/marginal_densities

        self.predicted_labels = posterior_probabilities.argmax(axis=0)

        return self.predicted_labels

    def calculate_error(self, ground_truth):

        bool_predictions = np.array(self.predicted_labels != ground_truth)

        gaussian_error = float(bool_predictions.sum()/ground_truth.shape[0])

        return gaussian_error * 100

    def calculate_accuracy(self, ground_truth):
        boolean_predictions = np.array(self.predicted_labels == ground_truth)

        accuracy = float(boolean_predictions.sum() / ground_truth.shape[0])

        return accuracy * 100


if __name__ == '__main__':
    Data, Label = load_iris()

    (DTR, LTR), (DTE, LTE) = split_db_2to1(Data, Label)

    models = ["Multivariate", "Naive Bayes", "Tied", "Tied Naive Bayes"]

    for index, model in enumerate(models):

        error = leave_one_out_cross_validation(GaussianClassifier, model, Data, Label)
        print(f"The error rate of the prediction of model {model} : {error:.1f}%")
        print("\n---------------------------------\n")
        # print(LOO_Cross_Validation(Gaussian_Classifier,model,Data,Label)[1].T)

    # log_test = np.load("Dataset/LOO_logSJoint_MVG.npy")
    # log_test = np.load("Dataset/LOO_logSJoint_NaiveBayes.npy")
    # log_test = np.load("Dataset/LOO_logSJoint_TiedMVG.npy")
    # log_test = np.load("Dataset/LOO_logSJoint_TiedNaiveBayes.npy")
    # print(log_test)
