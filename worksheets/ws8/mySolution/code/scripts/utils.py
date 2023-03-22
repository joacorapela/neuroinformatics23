import numpy as np

def sim_data(T, p, q):
    ''' simulate stimulus X and neural firing Y for T trials.
    Stimulus is on with probabiilty p, neuron active with probability q,
    independent
    returns a 2x2 array of bin counts for all possibilities'''

    X = np.random.rand(T)<p
    Y = np.random.rand(T)<q
    C = np.bincount(X+Y*2, minlength=4).reshape((2,2))/T
    return C


def H(X): # entropy of a probability distribution (any shape array)
    X = X.ravel()
    nonzero = (X>0)
    return -(X[nonzero]*np.log2(X[nonzero])).sum()


def mi(C): # mutual information of a joint distribution (2d array)
    H0 = H(C.sum(0))
    H1 = H(C.sum(1))
    H01 = H(C)
    #print(H0, H1, H01) uncomment this line to debug
    return H0+H1-H01

def H_cv(X1,X2):
    '''cross-validated entropy with training set probability distribution
    and test set distribution X2'''
    X1 = X1.ravel()
    X2 = X2.ravel()
    nonzero = (X2>0)
    # nonzero = (X1>0)
    answer = -(X2[nonzero]*np.log2(X1[nonzero])).sum()
    return answer

def mi_cv(C1,C2):
    '''cross-validated mutual information with training set probability
    distribution X1 and test set distribution X2'''
    H0 = H_cv(C1.sum(0), C2.sum(0))
    H1 = H_cv(C1.sum(1), C2.sum(1))
    H01 = H_cv(C1, C2)
    #print(H0, H1, H01) uncomment this line to debug
    return H0+H1-H01

