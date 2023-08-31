
from Utils import *


class BinaryTaskEvaluator:

    def __init__(self, prior, cfn, cfp):
        self.prior = prior
        self.cfn = cfn
        self.cfp = cfp
        self.llr = None
        self.ground_truth = None
        self.confusion_matrix = None
        self.fnr = None
        self.fpr = None

    def compute_bayes_risk(self, llr, ground_truth):

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

        plt.plot(1 - self.fnr, self.fpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.show()
