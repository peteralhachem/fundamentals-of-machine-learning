
from Utils import load_iris_function, split_db_2to1

if __name__ == '__main__':

    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)
