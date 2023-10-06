import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from src.mvg import multivariate_gaussian
import os


# ---Calculate the mean--- #
def calculate_mean(data_matrix):
    """
    Calculate the mean of the data matrix.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :return: mean of the data matrix.
    """
    return data_matrix.mean(1).reshape(data_matrix.shape[0], 1)


# ---Center the data points--- #

def center_data(data_matrix):
    """
    Center the data points by removing the mean.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :return: centered data points.

    """
    centered_data = data_matrix - calculate_mean(data_matrix)
    return centered_data


# ---Calculate the covariance matrix of the whole dataset--- #

def calculate_covariance(data_matrix, row_axis=True):
    """
    Calculates the covariance matrix of the whole dataset.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :param row_axis: perform the covariance in a row-major order.
    :return: Covariance matrix of the whole dataset of the shape (D,D)
    """
    if row_axis:
        return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[1]
    else:
        return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[0]


# ---Calculate the mean for each individual class--- #
def calculate_class_means(data_matrix, labels):
    """
    Calculates the mean for each class of the data matrix.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :param labels: labels associated with each data point of the data matrix.
    :return: array of mean for each class of the dataset (K,D,1).
    """
    mu = []

    for value in np.unique(labels):
        mu.append(calculate_mean(data_matrix[:, labels == value]))

    return np.array(mu)


# ---Compute the covariance matrix of each individual class--- #

def calculate_class_covariances(data_matrix, labels):
    """
    Calculates the covariance for each class of the data matrix.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :param labels: labels associated with each data point of the data matrix.
    :return: array of mean for each class of the dataset (K,D,D).

    """
    cov = []

    for value_label in np.unique(labels):
        cov.append(calculate_covariance(center_data(data_matrix[:, labels == value_label])))

    return np.array(cov)


# ---Calculate the class likelihood---#

def calculate_class_likelihood(mode, data_matrix, mu, cov):
    """
    Calculates the covariance for each class of the data matrix.
    :param mode:
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :param mu: mean of the dataset.
    :param cov: covariance matrix of the dataset.
    :return: array of scores for each class of the dataset (K,N).

    """
    score = []

    if mode == "Tied" or mode == "Tied Naive Bayes":
        for i in range(mu.shape[0]):
            densities = np.exp(multivariate_gaussian(data_matrix, mu[i], cov))
            score.append(densities)

    else:
        for i in range(mu.shape[0]):
            densities = np.exp(multivariate_gaussian(data_matrix, mu[i], cov[i]))
            score.append(densities)

    return np.array(score)


# ---Compute the confusion matrix between the predictions and the ground truth---#


def compute_confusion_matrix(predictions, ground_truth):
    """
    Compute the confusion matrix between the predictions and the ground truth.
    :param predictions: Predicted labels of the dataset.
    :param ground_truth: True labels of the dataset.
    :return: a DXD confusion matrix that represents the well-classified and misclassified datapoints from the test set.

    """
    confusion_matrix = np.zeros((len(np.unique(ground_truth)), len(np.unique(ground_truth))))

    occurrence = 1
    for i in range(predictions.shape[0]):
        confusion_matrix[predictions[i], ground_truth[i]] += occurrence

    return confusion_matrix


# ---------------------------------------------------------------------------------------------------------------------#


# -----Function to load the data from the iris csv-----#
def load_data_from_file(filename):
    # ----Initialize the variables of data and labels---- #
    data = []
    label = []

    # ----Create a dictionary to associate the labels with values----#
    labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    # ----File Handling----#

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # ----Create a 4x1 vector of the values of type float64 from each row of the file----#
            vector = np.array(row[:4], dtype=np.float64).reshape(4, 1)
            data.append(vector)
            # ----Add the numerical value associated to the string label----#
            label.append(labels[row[4]])

    data = np.array(data)

    # ---Concatenate the vectors---#
    data = np.hstack(data)

    label = np.array(label)

    return data, label


# ---Load Iris dataset using the sklearn---#
def load_iris_function():
    """
    Load Iris dataset from the IRIS package.
    :return: data_matrix and label matrix of the IRIS dataset.
    """
    data_matrix, labels = load_iris()['data'].T, load_iris()["target"]
    return data_matrix, labels


# ---Changing iris into a binary classification by Removing Setosa---#
def load_iris_binary():
    """
    Load 2 classes of the IRIS dataset from the IRIS package.
    :return: data_matrix and label matrix of the IRIS dataset.

    """
    data_matrix, labels = load_iris()['data'].T, load_iris()['target']

    data_matrix = data_matrix[:, labels != 0]  # We remove setosa from D

    labels = labels[labels != 0]  # We remove setosa from L
    labels[labels == 2] = 0  # We assign label 0 to virginica (was label 2)

    return data_matrix, labels


# ---Splitting the dataset into training and test sets---#

def split_db_2to1(data_matrix, labels, seed=0):
    """
    Split the dataset and label set into a training and test set.
    :param data_matrix: (D,N) matrix where D is the number of features and N is the number of samples.
    :param labels: labels associated with each data point of the data matrix.
    :param seed: control permutation for the random function.
    :return: training set and test set.

    """
    train_samples = int(data_matrix.shape[1] * (2.0 / 3.0))
    np.random.seed(seed)
    idx = np.random.permutation(data_matrix.shape[1])

    train_index = idx[:train_samples]
    test_index = idx[train_samples:]

    training_data = data_matrix[:, train_index]
    test_data = data_matrix[:, test_index]

    training_labels = labels[train_index]
    test_labels = labels[test_index]

    return (training_data, training_labels), (test_data, test_labels)


def kfold_cross_validation(classifiers, data_matrix, labels, k):
    """
    Kfold cross validation consists of splitting the dataset into K-Folds and using K-1 Folds for training the dataset
    and 1-fold for evaluating the dataset and computing the error, the final error is the cumulative sum of all the
    errors at each iteration.
    :param classifiers: list of defined classifiers to perform cross validation on.
    :param data_matrix: the dataset to use for splitting between training and evaluation set it has an initial shape of
    (D,N) where D is the number of features and N is the number of samples.
    :param labels: The labels associated to each dataset entry point having the shape (N,) where N is the number of
    samples.
    :param k: The number of splits/folds to be performed on the dataset.

    """

    final_error = np.zeros((len(classifiers)))
    data_split = np.array_split(data_matrix, k, axis=1)
    label_split = np.array_split(labels, k)

    for index, classifier in enumerate(classifiers):
        for i in range(len(data_split)):
            data_test = data_split[i]
            data_train = [element for element in data_split if element is not data_test]
            data_train = np.hstack(data_train)

            label_test = label_split[i]
            label_train = [element for element in label_split if element is not label_test]
            label_train = np.hstack(label_train)

            classifier.fit(data_train, label_train)
            classifier.predict(data_test)
            error = classifier.calculate_error(label_test)
            final_error[index] += error

            # --- Visualize the error at each step of the K-fold--- #

            # print(f"K = {i}, error rate = {final_error[index]:.1f}%")
        print(f"K = {k}, error rate = {final_error[index]:.1f}%")


def leave_one_out_cross_validation(classifiers, data_matrix, labels):
    """
    Leave-one-out cross validation consists of splitting the dataset into 2 splits, 1 data point for evaluating
    the dataset and the rest for training the dataset and computing the error.
    The final error is the cumulative sum of all the errors at each iteration.
    :param classifiers: list of defined classifiers to perform cross validation on.
    :param data_matrix: the dataset to use for splitting between training and evaluation set it has an initial shape of
    (D,N) where D is the number of features and N is the number of samples.
    :param labels: The labels associated to each dataset entry point having the shape (N,) where N is the number of
    samples.

    """

    error = np.zeros(labels.shape[0])

    for classifier in classifiers:

        for i in range(data_matrix.shape[1]):
            train_data = np.delete(data_matrix, i, 1)
            train_label = np.delete(labels, i)
            test_data = data_matrix[:, i:i + 1]
            test_label = np.array(labels[i]).reshape(1, 1)

            classifier.fit(train_data, train_label)
            classifier.predict(test_data)
            error[i] = classifier.calculate_error(test_label) / data_matrix.shape[1]

        classifier.save_results(error.sum())


# ------------------------------------------------------------------------------#


# ---Plotting the data, important to note that this applies to the Iris dataset---#

def histogram_plot(data, label):
    """
        Histogram plot function used to analyse each class with each feature present in the dataset.
        For the IRIS dataset for example we have 3 classes that are present.

        :param data: data matrix used for the plot (D,N) shape where D is the dimension of the features
        and N is the number of samples.
        :param label: label matrix containing the associated labels for each data point (N,)
        where N is the number of samples.

    """

    attribute_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axs = plt.subplots(2, 2)
    n_bins = 10
    r_width = 1
    alpha = 0.6
    data_0 = data[:, label == 0]
    data_1 = data[:, label == 1]
    data_2 = data[:, label == 2]

    for i in range(len(np.unique(label)) - 1):
        for j in range(len(np.unique(label)) - 1):
            axs[i, j].hist(data_0[int(str(i) + str(j), 2)], bins=n_bins, rwidth=r_width, alpha=alpha, density=True)
            axs[i, j].hist(data_1[int(str(i) + str(j), 2)], bins=n_bins, rwidth=r_width, alpha=alpha, density=True)
            axs[i, j].hist(data_2[int(str(i) + str(j), 2)], bins=n_bins, rwidth=r_width, alpha=alpha, density=True)

            axs[i, j].set_xlabel(attribute_names[int(str(i) + str(j), 2)])

    fig.tight_layout(pad=2.0)
    if os.path.exists("../img"):
        fig.savefig("../img/features_histogram.png")

    else:
        os.mkdir("../img")
        fig.savefig("../img/features_histogram.png")
    plt.show()


def pca_lda_scatter_plot(data, label, filename):
    """
    Scatter plot function mainly used to display the values after performing PCA or LDA.
    For the IRIS dataset for example we have 3 classes that are present.

    :param data: data matrix used for the plot (D,N) shape where D is the dimension of the features
    and N is the number of samples.
    :param label: label matrix containing the associated labels for each data point (N,)
    where N is the number of samples.
    :param filename: name to save the image.

    """

    data_0 = data[:, label == 0]
    data_1 = data[:, label == 1]
    data_2 = data[:, label == 2]

    for index_1 in range(data.shape[0]):
        for index_2 in range(data.shape[0]):
            if index_1 == index_2:
                continue

            elif index_1 < index_2:

                plt.scatter(data_0[index_1], data_0[index_2], label="Setosa", c="blue")
                plt.scatter(data_1[index_1], data_1[index_2], label="Versicolor", c="orange")
                plt.scatter(data_2[index_1], data_2[index_2], label="Virginica", c="green")
                plt.legend()

            else:
                continue

    plt.tight_layout()

    if os.path.exists("../img"):
        plt.savefig(f"../img/{filename}.png")

    else:
        os.mkdir("../img")
        plt.savefig(f"../img/{filename}.png")

    plt.show()


# ------------------------Bayes Model Evaluation----------------------------------------#
def bayes_error_plot(eff_prior_log_odds, min_dcf_values: dict, normalized_dcf_values: dict):

    """
    Plot the Bayes error using diverse effective prior log-odds which calculates the priors.
    :param eff_prior_log_odds: array of integers used to calculate the effective prior.
    :param min_dcf_values: dictionary containing the minimum DCF of all the classifiers computed.
    :param normalized_dcf_values: dictionary containing the normalized DCF of all the classifiers computed.

    """

    for key in min_dcf_values.keys():
        plt.plot(eff_prior_log_odds, normalized_dcf_values[key], label=f'DCF({key})')
        plt.plot(eff_prior_log_odds, min_dcf_values[key], label=f'min_DCF({key})')

    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('Prior log-odds')
    plt.ylabel('DCF value')
    plt.legend(loc='lower left')

    plt.show()
    plt.savefig('../img/bayes_error_plot.png')
    plt.close()
