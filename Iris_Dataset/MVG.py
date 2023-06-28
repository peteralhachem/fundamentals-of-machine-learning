import numpy as np

def MVG(X,mu,C):

    Y =[]


    for value in range(X.shape[1]):

        constant = -0.5 * C.shape[0] * np.log(2 * np.pi)
        variable_1 = -0.5 * np.linalg.slogdet(C)[1]

        variable_2 = X[:,value:value+1] - mu
        variable_3 = -0.5 * np.dot(variable_2.T,np.dot(np.linalg.inv(C),variable_2))

        Y.append(constant+variable_1+variable_3)

    return np.array(Y).ravel()

def Log_Likelihood_Estimator(X,mu,C):

    Y = MVG(X,mu,C)
    return np.sum(Y)








