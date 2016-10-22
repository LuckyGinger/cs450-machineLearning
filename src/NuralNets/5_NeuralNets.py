import random
import bunch
from sklearn import preprocessing
import numpy as np
import pandas as pd
from learndata import LearnData as ld
from learndata import ManipulateData as md



class Net:
    def __init__(self, data, target, num_neurons=[1], bias=-1):
        self._num_inputs = len(data[0])  # const value
        self._num_neurons = num_neurons  # const value
        self.data = data
        self.bias = bias
        self.inputs = []
        self.neurons = []
        self.outputs = []

        self._create_neurons()
        # self.calculate_outputs()

        # print(len(self.outputs))
        # print(self.outputs)

        for i, x in enumerate(self.neurons):
            print("Layer #" + str(i) + ":")
            for j, y in enumerate(x):
                print("\tNeuron #" + str(j) + ": " + str(y))

        # for i in range(self._num_neurons):
        #     print(self.neurons[i])

        return

    def _create_neurons(self):
        for x, j in enumerate(self._num_neurons):  # [10, 2]
            self.neurons.append([])  # To create j number of layers
            for i in range(j):  # To create i number of nodes in j layers
                if x == 0:  # If it's the first layer of Neurons after the inputs
                    self.neurons[x].append(Neuron(self._num_inputs))
                    print("F")
                else:
                    self.neurons[x].append(Neuron(self._num_neurons[x - 1]))
                    print("S")
        return

    # def calculate_outputs(self):
    #     temp_data = [np.insert(i, 0, self.bias) for i in self.data]  # add the bias to the beginning of each row
    #     # print(temp_data)
    #
    #     for data in temp_data:
    #         total = np.zeros(self._num_neurons)
    #         for index_neuron, neuron in enumerate(self.neurons):
    #             for index_weight, weight in enumerate(neuron.get_weights()):
    #                 total[index_neuron] += data[index_weight] * weight
    #
    #         for i, x in enumerate(total):
    #             if x > 0:
    #                 total[i] = 1
    #             elif x <= 0:
    #                 total[i] = 0
    #         self.outputs.append(total)
    #     return


class Neuron:
    def __init__(self, num_inputs=1):
        self.weights = []
        self._create_neuron(num_inputs)
        return

    def __str__(self):
        return str(self.get_weights())

    def append_weight(self, weight):
        self.weights.append(weight)

    def get_weights(self):
        return self.weights

    def get_weight(self, index):
        return self.weights[index]

    def _create_neuron(self, num_inputs):
        self.append_weight(round(random.uniform(-1, 1), 2))  # bias weight
        for _ in range(num_inputs):
            self.append_weight(round(random.uniform(-1, 1), 2))  # set the neuron weight here
        pass


def main():
    split = 0.7
    file = pd.read_csv('test_data.csv').values
    print(file[0])
    inputs = len(file[0])
    num_neurons = [10, 2, 7, 13, 2]  # array for the number of neurons in each hidden layer
    X_data, X_target, y_data, y_target = md.split_data(file, split)
    print(X_data)
    Net(X_data, X_target, num_neurons=num_neurons)

    # print("Irises Data:")
    # iris = ld.load_dataset('iris')
    # X_data, X_target, y_data, y_target = md.split_data(iris, split)
    # for i in X_data:
    #     print(i)
    # Net(X_data, X_target, num_neurons=num_neurons)
    #
    #
    # print("Pima Indians Data:")
    # pima = ld.load_dataset('pima')
    # X_data, X_target, y_data, y_target = md.split_data(pima, split)
    # for i in X_data:
    #     print(i)
    # Net(X_data, X_target, num_neurons=num_neurons)
if __name__ == '__main__':
    main()
