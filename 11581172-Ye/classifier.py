import numpy as np

# Implementation for a binary classifier with perceptron with a fixed learning rate
class BinaryPerceptron:
    def __init__(self):
        self.weight = None
        # fix the learning rate τ = 1
        self.learningRate = 1
        self.mistakeList = []
        self.accuracyList = []

    def fit(self, X, y, iters):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                # make prediction
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the weight: w = w + τ • y_t • x_t
                    self.weight += self.learningRate * y_t * x_t
                    mistake += 1
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Binary Perceptron Iteration #' + str(_) + ' num of mistake is:' + str(mistake) + ' and the Accuracy is:' + '{:.4f}'.format(accuracy))

        return self

    def predict(self, X):
        # predict using current weight
        y_ht = np.dot(X, self.weight)

        return np.where(y_ht >= 0, 1, -1)

    def accuracy(self, X, y):
        y_ht = self.predict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy

# Implementation for a binary classifier with PA with a changed learning rate
class BinaryPA:
    def __init__(self):
        self.weight = None
        self.learningRate = 0
        self.mistakeList = []
        self.accuracyList = []
    def fit(self, X, y, iters):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                # make prediction
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the learning rate: τ = (1-(y_t)•(w•x_t))/||x||^2
                    self.learningRate = (1-y_t*np.dot(self.weight, x_t))/(np.linalg.norm(x_t)**2)
                    # update the weight: w = w + τ • y_t • x_t
                    self.weight += self.learningRate * y_t * x_t

                    mistake += 1
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Binary PA Iteration #' + str(_) + ' num of mistake is:' + str(mistake) +  ' and the Accuracy is:' + '{:.4f}'.format(accuracy))

        return self

    def predict(self, X):
        # predict using current weight
        y_ht = np.dot(X, self.weight)

        return np.where(y_ht >= 0, 1, -1)

    def accuracy(self, X, y):
        y_ht = self.predict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy

# Implementation for Multi-class online learning algorithm with Perceptron
class MultiClassPerceptron:
    def __init__(self):
        self.weight = None
        self.classes = 10
        # fix the learning rate τ = 1
        self.learningRate = 1
        self.mistakeList = []
        self.accuracyList = []

    def fit(self, X, y, iters):
        n_samples, n_features = X.shape
        self.weight = np.zeros([self.classes, n_features])
        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the weight correct and incorrect class
                    self.weight[y_t] += self.learningRate * x_t
                    self.weight[y_ht] -= self.learningRate * x_t
                    mistake += 1
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Multi-class Perceptron Iteration #' + str(_) + ' num of mistake is:' + str(mistake) +  ' and the Accuracy is:' + '{:.4f}'.format(accuracy))
        return self

    def predict(self, X):
        # predict using current weight
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_ht = np.dot(X, self.weight.T)

        return np.argmax(y_ht, axis=1)

    def accuracy(self, X, y):
        y_ht = self.predict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy

# Implementation for Multi-class online learning algorithm with PA
class MultiClassPA:
    def __init__(self):
        self.weight = None
        self.classes = 10
        self.mistakeList = []
        self.accuracyList = []

    def fit(self, X, y, iters):

        n_samples, n_features = X.shape
        self.weight = np.zeros([self.classes, n_features])
        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the learning rate: τ = (1-w • F(x_t, y_t)-w • F(x_t, y_ht))/(||F(x_t, y_t)- F(x_t, y_ht)||**2)
                    self.learningRate = (1-np.dot(self.weight[y_t], x_t)-np.dot(self.weight[y_ht], x_t))/(np.linalg.norm(self.weight[y_t]-self.weight[y_ht])**2)
                    # update the weight correct and incorrect class
                    self.weight[y_t] += self.learningRate * x_t
                    self.weight[y_ht] -= self.learningRate * x_t
                    mistake += 1
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Multi-class PA Iteration #' + str(_) + ' num of mistake is:' + str(mistake) +  ' and the Accuracy is:' + '{:.4f}'.format(accuracy))
        return self

    def predict(self, X):
        # predict using current weight
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_ht = np.dot(X, self.weight.T)

        return np.argmax(y_ht, axis=1)

    def accuracy(self, X, y):
        y_ht = self.predict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy

# Implementation for Averaged Binary Perceptron
class AveragedBinaryPerceptron:
    def __init__(self):
        self.weight = None
        self.averagedWeight = None
        # fix the learning rate τ = 1
        self.learningRate = 1
        self.mistakeList = []
        self.accuracyList = []

    def fit(self, X, y, iters):
        n_samples, n_features = X.shape
        # initialize the weight and averaged weight
        self.weight = np.zeros(n_features)
        self.averagedWeight = np.zeros(n_features)

        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the weight: w = w + τ • y_t • x_t
                    self.weight += self.learningRate * y_t * x_t
                    mistake += 1
                self.averagedWeight += self.weight
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Averaged Binary Perceptron Iteration #' + str(_) + ' num of mistake is:' + str(mistake) +  ' and the Accuracy is:' + '{:.4f}'.format(accuracy))
        return self

    def predict(self, x_t):
        # predict using current weight
        y_ht = np.dot(x_t, self.weight)

        return np.where(y_ht >= 0, 1, -1)
    
    def avgPredict(self, X):
        # predict using averaged weight
        y_ht = np.dot(X, self.averagedWeight)

        return np.where(y_ht >= 0, 1, -1)

    def accuracy(self, X, y):

        y_ht = self.avgPredict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy

class AveragedMultiClassPerceptron:
    def __init__(self):
        self.weight = None
        self.averagedWeight = None
        self.classes = 10
        # fix the learning rate τ = 1
        self.learningRate = 1
        self.mistakeList = []
        self.accuracyList = []

    def fit(self, X, y, iters):
        n_samples, n_features = X.shape
        # initialize the weight and averaged weight
        self.weight = np.zeros([self.classes, n_features])
        self.averagedWeight = np.zeros([self.classes, n_features])

        for _ in range(iters):
            mistake = 0
            for x_t, y_t in zip(X,y):
                y_ht = self.predict(x_t)
                if y_ht != y_t:
                    # update the weight correct and incorrect class
                    self.weight[y_t] += self.learningRate * x_t
                    self.weight[y_ht] -= self.learningRate * x_t
                    mistake += 1
                self.averagedWeight += self.weight
            self.mistakeList.append(mistake)
            accuracy = self.accuracy(X, y)
            self.accuracyList.append(accuracy)
            print('Averaged Binary Perceptron Iteration #' + str(_) + ' num of mistake is:' + str(mistake) +  ' and the Accuracy is:' + '{:.4f}'.format(accuracy))
        return self

    def predict(self, x_t):
        # predict using current weight
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
        y_ht = np.dot(x_t, self.weight)

        return np.argmax(y_ht, axis=1)
    
    def avgPredict(self, X):
        # predict using averaged weight
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_ht = np.dot(X, self.averagedWeight.T)

        return np.argmax(y_ht, axis=1)


    def accuracy(self, X, y):

        y_ht = self.avgPredict(X)
        accuracy = np.mean(y_ht == y)
        return accuracy



