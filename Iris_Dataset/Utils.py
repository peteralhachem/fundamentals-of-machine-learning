import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from MVG import multivariate_gaussian
from SVM import SVM
from Gaussian_Classifiers import GaussianClassifier
from Logistic_Regression import LogisticRegression


# ---Calculate the mean--- #
def calculate_mean(data_matrix):
    return data_matrix.mean(1).reshape(data_matrix.shape[0], 1)


# ---Center the data points--- #

def center_data(data_matrix):
    centered_data = data_matrix - calculate_mean(data_matrix)
    return centered_data


# ---Calculate the covariance matrix of the whole dataset--- #

def calculate_covariance(data_matrix, row_axis=True):
    if row_axis:
        return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[1]
    else:
        return np.dot(data_matrix, data_matrix.T) / data_matrix.shape[0]


# ---Calculate the mean for each individual class--- #
def calculate_class_means(data_matrix, labels):
    mu = []

    for value in np.unique(labels):
        mu.append(calculate_mean(data_matrix[:, labels == value]))

    return np.array(mu)


# ---Compute the covariance matrix of each individual class--- #

def calculate_class_covariances(data_matrix, labels):
    cov = []

    for value_label in np.unique(labels):
        cov.append(calculate_covariance(center_data(data_matrix[:, labels == value_label])))

    return np.array(cov)


# ---Calculate the class likelihood---#

def calculate_class_likelihood(mode, data_matrix, mu, cov):
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
def load_iris():
    data_matrix, labels = load_iris()['data'].T, load_iris()["target"]
    return data_matrix, labels


# ---Changing iris into a binary classification by Removing Setosa---#
def load_iris_binary():
    data_matrix, labels = load_iris()['data'].T, load_iris()['target']

    data_matrix = data_matrix[:, labels != 0]  # We remove setosa from D

    labels = labels[labels != 0]  # We remove setosa from L
    labels[labels == 2] = 0  # We assign label 0 to virginica (was label 2)

    return data_matrix, labels


# ---Splitting the dataset into training and test sets---#

def split_db_2to1(data_matrix, labels, seed=0):
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


def kfold_cross_validation(classifier, model, data_matrix, labels, k):
    final_error = 0.0
    data_split = np.split(data_matrix, labels, 1)
    label_split = np.split(labels, k)
    gc = classifier(model)

    for i in range(len(data_split)):
        data_test = data_split[i]
        data_train = np.delete(data_split, i, 0)

        label_test = label_split[i]
        label_train = np.delete(label_split, i, 0)

        for j in range(len(data_split) - 1):
            gc.fit(data_train[j], label_train[j])
            predicted_labels = gc.predict(data_test)
            final_error += gc.calculate_error(label_test)

    return final_error


def leave_one_out_cross_validation(classifier, model, data_matrix, labels, lambda_value=None, k=None, c=None,
                                   gamma=None, constant=None,
                                   degree=None):
    error = np.zeros(labels.shape[0])

    classifier = classifier(model)

    for i in range(data_matrix.shape[1]):
        train_data = np.delete(data_matrix, i, 1)
        train_label = np.delete(labels, i)
        test_data = data_matrix[:, i:i + 1]
        test_label = np.array(labels[i]).reshape(1, 1)

        if isinstance(classifier, GaussianClassifier):

            classifier.fit(train_data, train_label)
            predicted_labels = classifier.predict(test_data)
            error[i] = classifier.calculate_error(test_label) / data_matrix.shape[1]

        elif isinstance(classifier, LogisticRegression):

            j = classifier.fit(train_data, train_label, value_lambda=lambda_value)
            predicted_labels = classifier.predict(test_data)
            error[i] = classifier.calculate_error(test_label) / data_matrix.shape[1]

        elif isinstance(classifier, SVM):

            if model == "Linear":
                classifier.fit(train_data, train_label, k, c)

            elif model == "Kernel Polynomial":
                classifier.fit(train_data, train_label, k, c, constant=constant, degree=degree)

            elif model == "Kernel RBF":
                classifier.fit(train_data, train_label, k, c, gamma=gamma)

            predicted_labels = classifier.Predict(test_data)

            error[i] = classifier.calculate_error(test_label) / data_matrix.shape[1]

    return error.sum()


# ------------------------------------------------------------------------------#


# ---Plotting the data, important to note that this applies to the Iris dataset---#

def plot(data, label):
    attribute_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, axs = plt.subplots(2, 2)

    data_0 = data[:, label == 0]
    data_1 = data[:, label == 1]
    data_2 = data[:, label == 2]

    for i in range(len(np.unique(label)) - 1):
        for j in range(len(np.unique(label)) - 1):
            axs[i, j].hist(data_0[int(str(i) + str(j), 2)], bins=10, alpha=0.6, rwidth=0.8, density=True)
            axs[i, j].hist(data_1[int(str(i) + str(j), 2)], bins=10, alpha=0.6, rwidth=0.8, density=True)
            axs[i, j].hist(data_2[int(str(i) + str(j), 2)], bins=10, alpha=0.6, rwidth=0.8, density=True)

            axs[i, j].set_xlabel(attribute_names[int(str(i) + str(j), 2)])

    fig.tight_layout(pad=2.0)
    plt.legend()
    plt.show()


def scatter_plot(data, label):
    data_0 = data[:, label == 0]
    data_1 = data[:, label == 1]
    data_2 = data[:, label == 2]

    attribute_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    for index_1 in range(data.shape[0]):
        for index_2 in range(data.shape[0]):
            if index_1 == index_2:
                continue
            plt.figure()
            plt.xlabel(attribute_names[index_1])
            plt.ylabel(attribute_names[index_2])
            plt.scatter(data_0[index_1], data_0[index_2], label="Setosa")
            plt.scatter(data_1[index_1], data_1[index_2], label="Versicolor")
            plt.scatter(data_2[index_1], data_2[index_2], label="Virginica")

            plt.legend()
            plt.tight_layout()

    plt.show()


# ------------------------Bayes Model Evaluation----------------------------------------#
def bayes_error_plot(eff_prior_log_odds, normalized_dcf, min_dcf, normalized_dcf_1=np.array([None]),
                     min_dcf_1=np.array([None])):
    plt.plot(eff_prior_log_odds, normalized_dcf, label='DCF (e = 0.001)', color='r')
    plt.plot(eff_prior_log_odds, min_dcf, label='minDCF (e = 0.001)', color='b')

    if normalized_dcf_1.any() and min_dcf_1.any():
        plt.plot(eff_prior_log_odds, normalized_dcf_1, label='DCF (e = 1)', color='y')
        plt.plot(eff_prior_log_odds, min_dcf_1, label='minDCF (e = 1)', color='g')

    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('Prior log-odds')
    plt.ylabel('DCF value')
    plt.legend(loc='lower left')

    plt.show()


def plot_density(data_matrix, marginal_log_density):
    x = np.linspace(-10, 5, 1000)

    plt.hist(data_matrix, bins=60, rwidth=0.9, density=True)
    plt.plot(x, marginal_log_density)
    plt.xlim([-10.0, 5.0])
    plt.ylim([0.0, 0.3])
    plt.show()
