from src.utils import *
from src.svm import SVM

if __name__ == '__main__':

    data_matrix, labels = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    models = ["Linear", "Polynomial", "RBF"]

    for model in models:
        if model == "Linear":
            K = [1]
            C = [0.1, 1, 10]
            for k in K:
                for c in C:
                    svm = SVM(mode=model, k=k, c=c)
                    svm.fit(DTR, LTR)
                    svm.predict(DTE)
                    svm.calculate_error(LTE)
                    svm.save_results()

        elif model == "Polynomial":
            Constant = [0, 1]
            K = [0, 1]
            for constant in Constant:
                for k in K:
                    svm = SVM(mode=model, k=k, c=1, constant=constant, degree=2)
                    svm.fit(DTR, LTR)
                    svm.predict(DTE)
                    svm.calculate_error(LTE)
                    svm.save_results()

        elif model == "RBF":
            Gamma = [1, 10]
            K = [0, 1]

            for k in K:
                for gamma in Gamma:
                    svm = SVM(mode=model, k=k, c=1, gamma=gamma)
                    svm.fit(DTR, LTR)
                    svm.predict(DTE)
                    svm.calculate_error(LTE)
                    svm.save_results()
