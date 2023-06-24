import numpy as np
import csv
import matplotlib.pyplot as plt

#----Calculate the mean----#
def Calculate_mean(data):
    return data.mean(1).reshape(data.shape[0],1)

#---Center the data points---#

def Center_Data(data):
    centered_data = data - Calculate_mean(data)
    return centered_data


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


#---Plotting the data---#

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



if __name__ == '__main__':

    data,label = Load_Data_from_file('Dataset/iris.csv')

    scatter_plot(Center_Data(data),label)

