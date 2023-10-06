from src.utils import *
from src.binary_evaluation import BinaryTaskEvaluator
from src.multiclass_evaluation import MulticlassTaskEvaluator
if __name__ == '__main__':
    models = ["Binary", "Multiclass"]

    for model in models:
        if model == "Binary":
            eps = [0.001, 1]
            llr = None
            binary_labels = None
            for value in eps:
                if value == 0.001:
                    binary_labels = np.load('../dataset/commedia/commedia_labels_infpar.npy')
                    llr = np.load('../dataset/commedia/commedia_llr_infpar.npy')
                elif value == 1:
                    binary_labels = np.load('../dataset/commedia/commedia_labels_infpar_eps1.npy')
                    llr = np.load('../dataset/commedia/commedia_llr_infpar_eps1.npy')

                bte = BinaryTaskEvaluator(0.5, 1, 1)
                bte.compute_bayes_risk(llr, binary_labels)
                bte.compute_min_dcf(llr, binary_labels)
                bte.save_results(value)

                bte = BinaryTaskEvaluator(0.8, 1, 1)
                bte.compute_bayes_risk(llr, binary_labels)
                bte.compute_min_dcf(llr, binary_labels)
                bte.save_results(value)

                bte = BinaryTaskEvaluator(0.5, 10, 1)
                bte.compute_bayes_risk(llr, binary_labels)
                bte.compute_min_dcf(llr, binary_labels)
                bte.save_results(value)

                bte = BinaryTaskEvaluator(0.8, 1, 10)
                bte.compute_bayes_risk(llr, binary_labels)
                bte.compute_min_dcf(llr, binary_labels)
                bte.save_results(value)

        elif model == "Multiclass":
            eps = [0.001, 1]
            ll = None
            mlt_labels = None
            prior_vector = np.array([1/3, 1/3, 1/3])
            cost_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            for value in eps:
                if value == 0.001:
                    mlt_labels = np.load('../dataset/commedia/commedia_labels.npy')
                    ll = np.load('../dataset/commedia/commedia_ll.npy')
                elif value == 1:
                    mlt_labels = np.load('../dataset/commedia/commedia_labels_eps1.npy')
                    ll = np.load('../dataset/commedia/commedia_ll_eps1.npy')

                mlte = MulticlassTaskEvaluator(prior_vector, cost_matrix)
                mlte.compute_dcf(ll, mlt_labels)
                mlte.save_results(value)
