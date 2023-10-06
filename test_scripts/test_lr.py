from src.utils import *
from src.logistic_regression import LogisticRegression


if __name__ == '__main__':

    lambda_values = [10**(-6), 10**(-3), 10**(-1), 1]
    models = ["Binary", "Multiclass"]

    for model in models:
        if model == "Binary":
            data_matrix, labels = load_iris_binary()
            (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

        elif model == "Multiclass":
            data_matrix, labels = load_iris_function()
            (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

        else:
            print("Unsupported model, expected Binary or Multiclass.")

        for value in lambda_values:
            lr = LogisticRegression(model, value)
            lr.fit(DTR, LTR)
            lr.predict(DTE)
            error = lr.calculate_error(LTE)
            lr.save_results()
