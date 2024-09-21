import numpy as np
import mnist_reader
import matplotlib.pyplot as plt
from classifier import BinaryPerceptron
from classifier import BinaryPA
from classifier import MultiClassPerceptron
from classifier import MultiClassPA
from classifier import AveragedBinaryPerceptron
from classifier import AveragedMultiClassPerceptron
def main():

    # read the data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Convert the label to binary [1, -1] where [1,3,5,7,9] are [-1] and [0,2,4,6,8] are [1]
    y_trainBi = np.where(y_train % 2 == 0, 1, -1)
    y_testBi = np.where(y_test % 2 == 0, 1, -1)

    # Using the Binary Classifier with Perceptron
    binaryPerceptron = BinaryPerceptron()
    binaryPerceptron.fit(X_train, y_trainBi, 50)

    # Using the Binary Classifier with PA
    binaryPA = BinaryPA()
    binaryPA.fit(X_train, y_trainBi, iters = 50)
    # Plot the number of iterations with number of mistakes for Binary Perceptron and Binary PA
    plt.plot(binaryPerceptron.mistakeList, label='Binary Perceptron')
    plt.plot(binaryPA.mistakeList, label='Binary PA')
    plt.title('Number of iterations with number of mistakes for Binary Perceptron and Binary PA')
    plt.xlabel('Number of iterations')
    plt.ylabel('Number of mistakes')
    plt.legend()
    plt.show()

    # Plot accuracy with iteration for Binary Perceptron and Binary PA
    binaryPerceptron_train = BinaryPerceptron()
    binaryPerceptron_train.fit(X_train, y_trainBi, 20)
    binaryPerceptron_test = BinaryPerceptron()
    binaryPerceptron_test.fit(X_test, y_testBi, 20)

    binaryPA_train = BinaryPA()
    binaryPA_train.fit(X_train, y_trainBi, iters = 20)
    binaryPA_test = BinaryPA()
    binaryPA_test.fit(X_test, y_testBi, iters = 20)


    plt.plot(binaryPerceptron_train.accuracyList, label='Binary Perceptron Training Accuracy')
    plt.plot(binaryPerceptron_test.accuracyList, label='Binary Perceptron Testing Accuracy')
    plt.plot(binaryPA_train.accuracyList, label='Binary PA Training Accuracy')
    plt.plot(binaryPA_test.accuracyList, label='Binary PA Testing Accuracy')
    plt.title('Accuracy with iteration for training and testing for Binary Perceptron and PA')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot accuracy with iteration for Binary Perceptron and Averaged Perceptron
    binaryPerceptron_train = BinaryPerceptron()
    binaryPerceptron_test = BinaryPerceptron()
    avgBiPerceptron_train = AveragedBinaryPerceptron()
    avgBiPerceptron_test = AveragedBinaryPerceptron()

    binaryPerceptron_train.fit(X_train, y_trainBi, 20)
    binaryPerceptron_test.fit(X_test, y_testBi, 20)
    avgBiPerceptron_train.fit(X_train, y_trainBi, 20)
    avgBiPerceptron_test.fit(X_test, y_testBi, 20)

    plt.plot(binaryPerceptron_train.accuracyList, label='Perceptron Training Accuracy')
    plt.plot(binaryPerceptron_test.accuracyList, label='Perceptron Testing Accuracy')
    plt.plot(avgBiPerceptron_train.accuracyList, label='Averaged Perceptron Training Accuracy')
    plt.plot(avgBiPerceptron_test.accuracyList, label='Averaged Perceptron Testing Accuracy')
    plt.title('Accuracy with iteration for Perceptron and Averaged Perceptron')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Using the Multi-class Classifier with Perceptron

    multiClassPerceptron = MultiClassPerceptron()
    multiClassPerceptron.fit(X_train, y_train, iters = 50)

    # Using the Multi-class Classifier with PA
    '''
    multiClassPA = MultiClassPA()
    multiClassPA.fit(X_train, y_train, iters = 50)
    '''

    # Plot the number of iterations with number of mistakes for Multi-class Perceptron and Multi-class PA
    plt.plot(multiClassPerceptron.mistakeList, label='Multi-class Perceptron')
    '''
    plt.plot(multiClassPA.mistakeList, label='Multi-class PA')
    '''
    plt.title('Number of iterations with number of mistakes for Multi-class Perceptron and Multi-class PA')
    plt.xlabel('Number of iterations')
    plt.ylabel('Number of mistakes')
    plt.legend()
    plt.show()

    # Plot accuracy with iteration for Multi-class Perceptron and Multi-class PA
    multiClassPerceptron_train = MultiClassPerceptron()
    multiClassPerceptron_train.fit(X_train, y_train, 20)
    multiClassPerceptron_test = MultiClassPerceptron()
    multiClassPerceptron_test.fit(X_test, y_test, 20)
    '''
    multiClassPA_train = MultiClassPA()
    multiClassPA_train.fit(X_train, y_train, iters = 20)
    multiClassPA_test = MultiClassPA()
    multiClassPA_test.fit(X_test, y_test, iters = 20)
    '''
    plt.plot(multiClassPerceptron_train.accuracyList, label='Multi-class Perceptron Training Accuracy')
    plt.plot(multiClassPerceptron_test.accuracyList, label='Multi-class Perceptron Testing Accuracy')
    '''
    plt.plot(multiClassPA_train.accuracyList, label='Multi-class PA Training Accuracy')
    plt.plot(multiClassPA_test.accuracyList, label='Multi-class PA Testing Accuracy')
    '''
    plt.title('Accuracy with iteration for training and testing for Multi-class Perceptron and PA')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot accuracy with iteration for Multi-class Perceptron and Averaged Perceptron
    '''
    multiClassPerceptron_train = MultiClassPerceptron()
    multiClassPerceptron_test = MultiClassPerceptron()

    avgMultiClassPerceptron_train = AveragedMultiClassPerceptron()
    avgMultiClassPerceptron_test = AveragedMultiClassPerceptron()

    multiClassPerceptron_train.fit(X_train, y_train, 20)

    multiClassPerceptron_test.fit(X_test, y_test, 20)
    avgMultiClassPerceptron_train.fit(X_train, y_train, 20)
    avgMultiClassPerceptron_test.fit(X_test, y_test, 20)


    plt.plot(multiClassPerceptron_train.accuracyList, label='Perceptron Training Accuracy')
    plt.plot(multiClassPerceptron_test.accuracyList, label='Perceptron Testing Accuracy')
    plt.plot(avgMultiClassPerceptron_train.accuracyList, label='Averaged Perceptron Training Accuracy')
    plt.plot(avgMultiClassPerceptron_test.accuracyList, label='Averaged Perceptron Testing Accuracy')
    plt.title('Accuracy with iteration for Multi-class Perceptron and Averaged Multi-class Perceptron')
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    '''
if __name__ == "__main__":

    main()