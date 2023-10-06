from src.utils import *


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
        self.min_dcf = None
        self.dcf_u = None
        self.normalized_dcf = None
        self.dcf_u_values = None
        self.normalized_dcf_values = None

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

        self.dcf_u = self.prior * self.cfn * self.fnr + (1 - self.prior) * self.cfp * self.fpr

        self.normalized_dcf = self.dcf_u / min(self.prior * self.cfn, (1 - self.prior) * self.cfp)

        return self.dcf_u, self.normalized_dcf

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

        self.normalized_dcf_values = np.zeros(thresholds.shape)
        self.dcf_u_values = np.zeros(thresholds.shape)
        self.fnr = np.zeros(thresholds.shape)
        self.fpr = np.zeros(thresholds.shape)

        for index, threshold in enumerate(thresholds):
            predictions = np.int32(self.llr > threshold)

            self.confusion_matrix = compute_confusion_matrix(predictions, self.ground_truth)

            self.fnr[index] = self.confusion_matrix[0, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
            self.fpr[index] = self.confusion_matrix[1, 0] / (self.confusion_matrix[1, 0] + self.confusion_matrix[0, 0])

            self.dcf_u_values[index] = self.prior * self.cfn * self.fnr[index] + \
                                       (1 - self.prior) * self.cfp * self.fpr[index]

            self.normalized_dcf_values[index] = self.dcf_u_values[index] / min(self.prior * self.cfn, (1 - self.prior) *
                                                                               self.cfp)

        self.min_dcf = min(self.normalized_dcf_values)

        return self.min_dcf

    def roc_curve(self):
        """
        ROC curve plots the frequency of False Positive and True Positive rates
        while we vary the value of the threshold.
        """

        plt.plot(self.fpr, 1 - self.fnr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.show()
        plt.savefig('../img/roc_curve.png')
        plt.close()

    def __str__(self, eps=None):

        string = f"eps: {eps}\n"
        string += f"(pi, cfn, cfp):({self.prior}, {self.cfn}, {self.cfp})    | DCF: {self.normalized_dcf:.3f} | " \
                  f"min_DCF: {self.min_dcf:.3f}"
        string += f"\n------------------------------------------------\n"

        return string

    def save_results(self, eps):
        """
        Save the results of the binary evaluator into a txt file.
        :param eps: epsilon parameter required for the evaluation.

        """

        if os.path.exists('../results/evaluation'):
            pass
        else:
            os.mkdir('../results/evaluation')

        if os.path.exists('../results/evaluation/binary_evaluation.txt'):
            with open('../results/evaluation/binary_evaluation.txt', 'a') as file:
                file.write(self.__str__(eps))
        else:
            try:
                with open('../results/evaluation/binary_evaluation.txt', 'a') as file:
                    file.write(self.__str__(eps))

            except FileNotFoundError:
                print('cannot create file.')
