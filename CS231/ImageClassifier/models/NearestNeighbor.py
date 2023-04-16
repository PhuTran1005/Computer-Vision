import numpy as np

class NearestNeighbor:
    """
    Implementation Nearest Neibor Classifier
    """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        X is NxD where each row is an example. 
        y is 1-D of size N
        """
        # The nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        X is NxD where each row is an example we wish to predict label for
        """
        num_test = X.shape[0]
        # let make sure that the output type match the input type
        Y_pred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop for all test rows
        for i in range(num_test):
            # Find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr-X[i, :]), axis=1)
            min_index = np.argmin(distances) # get the index with smallest distancec
            Y_pred[i] = self.ytr[min_index]
        print(Y_pred)
        return Y_pred
