import numpy as np

# Split a dataset into a train and test set
def train_test_split( x, y=np.array([]), train_size=0.7, test_size = 0.3, random_state = 42 ):
    x = np.array(x)
    if test_size: train_size = 1 - test_size
    train_size = int(train_size * len(x))
    # get shuffle indices 
    indices  = np.random.permutation(len(x)) # axis=0 by default
    train_ , test_= indices [: train_size], indices [train_size: ]
    x_train, x_test = x[train_], x[test_]
    if y.size:
        # check if x and y have the same size
        assert(len(x)==len(y))
        y = np.array(y)
        y_train, y_test = y[train_], y[test_]
        return x_train, x_test, y_train, y_test
    return x_train, x_test
