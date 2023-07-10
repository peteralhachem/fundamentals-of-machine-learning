import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn.datasets
from MVG import MVG


#----Calculate the mean----#
def Calculate_mean(data):
    return data.mean(1).reshape(data.shape[0],1)

#---Center the data points---#

def Center_Data(data):
    centered_data = data - Calculate_mean(data)
    return centered_data

#---Calculate the covariance matrix of the whole dataset---#

def Calculate_Covarariance_Matrix(data,row_axis =True):
    if row_axis:
        return np.dot(data,data.T) /data.shape[1]
    else:
        return np.dot(data,data.T)/data.shape[0]


#---Calculate the mean for each individual class---#
def Calculate_class_means(Data,Label):

    mu = []

    for value in np.unique(Label):
        mu.append(Calculate_mean(Data[:,Label == value]))


    return np.array(mu)

#---Compute the covariance matrix of each individual class---#

def Calculate_class_covariances(Data,Label):

    Cov = []

    for value in np.unique(Label):
        Cov.append(Calculate_Covarariance_Matrix(Center_Data(Data[:,Label == value])))

    return np.array(Cov)


#---Calculate the class likelihood---#

def Calculate_class_likelihood(Data,mu,C,mode):

    Score = []

    if mode == "Tied" or mode == "Tied Naive Bayes":
        for i in range(mu.shape[0]):
            densities = np.exp(MVG(Data, mu[i], C))
            Score.append(densities)

    else:
        for i in range(mu.shape[0]):
            densities = np.exp(MVG(Data, mu[i], C[i]))
            Score.append(densities)



    return np.array(Score)


#---Compute the confusion matrix between the predictions and the ground truth---#


def ComputeConfusionMatrix(Predictions,Ground_Truth):

    ConfusionMatrix = np.zeros((len(np.unique(Ground_Truth)), len(np.unique(Ground_Truth))))

    occurence = 1
    for i in range(Predictions.shape[0]):
        ConfusionMatrix[Predictions[i],Ground_Truth[i]] += occurence

    return ConfusionMatrix






#----------------------------------------------------------------------------------------#



#-----Function to load the data from the iris csv-----#
def Load_Data_from_file(filename):

    #----Initialize the variables of data and labels----#
    data = []
    label = []

    #----Create a dictionary to associate the labels with values----#
    Labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    #----File Handings----#
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #----Create a 4x1 vector of the values of type float64 from each row of the file----#
            vector = np.array(row [:4],dtype=np.float64).reshape(4,1)
            data.append(vector)
            #----Add the numerical value associated to the string label----#
            label.append(Labels[row[4]])


    data = np.array(data)

    #---Concatenate the vectors---#
    data = np.hstack(data)

    label = np.array(label)

    return data,label

#---Load Iris dataset using the sklearn---#
def Load_Iris():

    D,L = sklearn.datasets.load_iris()['data'].T,sklearn.datasets.load_iris()["target"]
    return D,L

#---Changing iris into a binary classification by Removing Setosa---#
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

#---Splitting the dataset into training and test sets---#

def split_db_2to1(D,L,seed=0):
    nTrain = int(D.shape[1]*(2.0/3.0))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:,idxTrain]
    DTE = D[:,idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR,LTR),(DTE,LTE)


def KFold_cross_validation(Classifier,model,Data,Label,K):

    final_error = 0.0
    data_split = np.split(Data,K, 1)
    label_split = np.split(Label, K)
    gc = Classifier(model)

    for i in range(len(data_split)):
        data_test = data_split[i]
        data_train = np.delete(data_split, i, 0)

        label_test = label_split[i]
        label_train = np.delete(label_split, i, 0)

        for j in range(len(data_split)-1):
            gc.fit(data_train[j],label_train[j])
            Predicted_labels = gc.predict(data_test)
            final_error += gc.calculate_error(label_test)

    return final_error

def LOO_Cross_Validation(Classifier,model,Data,Label):

    gc = Classifier(model)
    error = 0
    Log = []

    for i in range (Data.shape[1]):
        train_data =np.delete(Data,i,1)
        train_label = np.delete(Label,i)
        test_data = Data[:,i:i+1]
        test_label = np.array(Label[i]).reshape(1,1)

        gc.fit(train_data,train_label)
        Predicted_labels,log = gc.predict(test_data)
        error += gc.calculate_error(test_label)/Data.shape[1]
        Log.append(log)

    return error,np.array(Log)











#------------------------------------------------------------------------------#


#---Plotting the data, important to note that this applies to the Iris dataset---#

def plot(data, label):
    attribute_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig,axs = plt.subplots(2,2)

    D0 = data[:,label == 0]
    D1 = data[:,label == 1]
    D2 = data[:,label == 2]



    for i in range (len(np.unique(label))-1):
        for j in range(len(np.unique(label))-1):
            axs[i, j].hist(D0[int(str(i)+str(j),2)], bins=10, alpha=0.6, rwidth=0.8, density=True)
            axs[i, j].hist(D1[int(str(i)+str(j),2)], bins=10, alpha=0.6, rwidth=0.8, density=True)
            axs[i, j].hist(D2[int(str(i)+str(j),2)], bins=10, alpha=0.6, rwidth=0.8, density=True)

            axs[i, j].set_xlabel(attribute_names[int(str(i)+str(j),2)])

    fig.tight_layout(pad=2.0)
    plt.legend()
    plt.show()


def scatter_plot(data,label):

    D0 = data[:,label==0]
    D1 = data[:,label==1]
    D2 = data[:,label==2]

    attribute_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    for index_1 in range(data.shape[0]):
        for index_2 in range(data.shape[0]):
            if index_1 == index_2:
                continue
            plt.figure()
            plt.xlabel(attribute_names[index_1])
            plt.ylabel(attribute_names[index_2])
            plt.scatter(D0[index_1],D0[index_2] ,label = "Setosa")
            plt.scatter(D1[index_1],D1[index_2],label="Versicolor")
            plt.scatter(D2[index_1],D2[index_2],label="Virginica")

            plt.legend()
            plt.tight_layout()

    plt.show()

#------------------------Bayes Model Evaluation----------------------------------------#
def BayesErrorPlot(effPriorLogOdds,Normalized_DCF,min_DCF,Normalized_DCF_1 = np.array([None]),min_DCF_1 = np.array([None])):

    plt.plot(effPriorLogOdds, Normalized_DCF, label='DCF (e = 0.001)', color='r')
    plt.plot(effPriorLogOdds, min_DCF, label='minDCF (e = 0.001)', color='b')

    if Normalized_DCF_1.any() and min_DCF_1.any():
        plt.plot(effPriorLogOdds, Normalized_DCF_1, label='DCF (e = 1)', color='y')
        plt.plot(effPriorLogOdds, min_DCF_1, label='minDCF (e = 1)', color='g')

    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('Prior log-odds')
    plt.ylabel('DCF value')
    plt.legend(loc='lower left')

    plt.show()



if __name__ == '__main__':

    data,label = Load_Data_from_file('Dataset/iris.csv')

    print(Calculate_mean(data))

    #scatter_plot(Center_Data(data),label)

