
from utils import *


class BinaryTaskEvaluator:

    def __init__(self, prior, cfn, cfp):
        """
        Constructor for the BinaryTaskEvaluator class
        :param prior: could be effective prior or normal prior that is the probability of classifying a class and the
        other class.
        :param cfn: cost of classifying a data point as belonging to class 0 while it belongs to class 1.
        :param cfp: cost of classifying a data point as belonging to class 1 while it belongs to class 0.

        """
        self.prior = prior
        self.cfn = cfn
        self.cfp = cfp
        self.llr = None
        self.ground_truth = None
        self.confusion_matrix = None
        self.fnr = None
        self.fpr = None

    def compute_bayes_risk(self, llr, ground_truth):
        """
        :param llr: log-likelihood ratio between the binary classes: (likelihood of class 1) / (likelihood of class 0).
        :param ground_truth: true labels associated with the dataset used.
        :return: actual detection cost function, normalized dcf value by dividing it with the minimum of 2 values.

        """

        self.llr = llr
        self.ground_truth = ground_truth
        threshold = - np.log((self.prior * self.cfn) / ((1 - self.prior) * self.cfp))

        predictions = np.int32(self.llr > threshold)

        self.confusion_matrix = compute_confusion_matrix(predictions, self.ground_truth)

        self.fnr = self.confusion_matrix[0, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        self.fpr = self.confusion_matrix[1, 0] / (self.confusion_matrix[1, 0] + self.confusion_matrix[0, 0])

        dcf_u = self.prior * self.cfn * self.fnr + (1 - self.prior) * self.cfp * self.fpr

        normalized_dcf = dcf_u / min(self.prior * self.cfn, (1 - self.prior) * self.cfp)

        return dcf_u, normalized_dcf

    def compute_min_dcf(self, llr, ground_truth):
        """
        :param llr: log-likelihood ratio between the binary classes: (likelihood of class 1) / (likelihood of class 0).
        :param ground_truth: true labels associated with the dataset used.
        :return: min_dcf computed after score calibration by using each llr as a threshold and see the smallest value
        of all the possible dcf values computed.

        """
        self.llr = llr
        self.ground_truth = ground_truth

        thresholds = np.array(self.llr)
        thresholds.sort()
        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])

        normalized_dcf_values = np.zeros(thresholds.shape)
        self.fnr = np.zeros(thresholds.shape)
        self.fpr = np.zeros(thresholds.shape)

        for index, threshold in enumerate(thresholds):

            predictions = np.int32(self.llr > threshold)

            self.confusion_matrix = compute_confusion_matrix(predictions, self.ground_truth)

            self.fnr[index] = self.confusion_matrix[0, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
            self.fpr[index] = self.confusion_matrix[1, 0] / (self.confusion_matrix[1, 0] + self.confusion_matrix[0, 0])

            dcf_u = self.prior * self.cfn * self.fnr[index] + (1 - self.prior) * self.cfp * self.fpr[index]

            normalized_dcf_values[index] = dcf_u / min(self.prior * self.cfn, (1 - self.prior) * self.cfp)

        return min(normalized_dcf_values)

    def roc_curve(self):
        """
        ROC curve plots the frequency of False Positive and True Positive rates
        while we vary the value of the threshold.
        """

        plt.plot(1 - self.fnr, self.fpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.show()
