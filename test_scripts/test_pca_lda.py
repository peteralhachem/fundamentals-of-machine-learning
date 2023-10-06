
from src.utils import *
from src.pca import PCA
from src.lda import LDA


if __name__ == '__main__':
    data_matrix, labels = load_iris_function()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(data_matrix, labels)

    pca = PCA(n_components=2)
    pca.fit(data_matrix)
    transformed_data_pca = pca.transform(data_matrix)
    pca.save_results()

    pca_lda_scatter_plot(transformed_data_pca, labels, 'pca_scatter_plot_2d')

    lda = LDA(n_components=2)
    lda.fit(data_matrix, labels)
    transformed_data_lda = lda.transform(data_matrix)
    lda.save_results()

    pca_lda_scatter_plot(transformed_data_lda, labels, 'lda_scatter_plot_2d')
