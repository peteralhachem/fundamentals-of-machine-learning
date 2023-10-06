from src.utils import *
from src.binary_evaluation import BinaryTaskEvaluator

if __name__ == '__main__':
    binary_labels = np.load('../dataset/commedia/commedia_labels_infpar.npy')
    llr = np.load('../dataset/commedia/commedia_llr_infpar.npy')

    bte = BinaryTaskEvaluator(0.5, 1, 1)
    bte.compute_bayes_risk(llr, binary_labels)
    bte.compute_min_dcf(llr, binary_labels)
    bte.roc_curve()

    # ----------------------------------Bayes Error plot------------------------------------- #

    effPriorLogOdds = np.linspace(-3, 3, 21)
    eps = [0.001, 1]
    min_dcf_array = []
    normalized_dcf_array = []
    min_dcf_dict = {}
    normalized_dcf_dict = {}

    priors = 1/(1 + np.exp(- effPriorLogOdds))

    for value in eps:
        if value == 0.001:
            binary_labels = np.load('../dataset/commedia/commedia_labels_infpar.npy')
            llr = np.load('../dataset/commedia/commedia_llr_infpar.npy')
        elif value == 1:
            binary_labels = np.load('../dataset/commedia/commedia_labels_infpar_eps1.npy')
            llr = np.load('../dataset/commedia/commedia_llr_infpar_eps1.npy')

        for prior in priors:
            bte = BinaryTaskEvaluator(prior, 1, 1)
            normalized_dcf_array.append(bte.compute_bayes_risk(llr, binary_labels)[1])
            min_dcf_array.append(bte.compute_min_dcf(llr, binary_labels))

        normalized_dcf_dict[f"eps={value}"] = normalized_dcf_array
        min_dcf_dict[f"eps={value}"] = min_dcf_array
        normalized_dcf_array = []
        min_dcf_array = []

    bayes_error_plot(effPriorLogOdds, min_dcf_dict, normalized_dcf_dict)
