
from utils import *
from pca import PCA
# from lda import LDA
# from gaussian_classifier import GaussianClassifier


if __name__ == '__main__':

    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    pca = PCA(n_components=2)
    pca.fit(data_matrix)
    transformed_data = pca.transform(data_matrix)

    print(pca)
