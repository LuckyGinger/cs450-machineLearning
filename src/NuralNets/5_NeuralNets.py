import random
import bunch
from sklearn import preprocessing
import numpy as np
import pandas as pd


class Net:
    def __init__(self, num_inputs=1, num_neurons=1, bias=-1):
        self._num_inputs = num_inputs  # const value
        self._num_neurons = num_neurons  # const value
        self.bias = bias
        self.inputs = []
        self.neurons = []

        self._create_neurons()


        for i in range(self._num_neurons):
            print(self.neurons[i])

        return

    def _create_neurons(self):
        for j in range(self._num_neurons):
            self.neurons.append(Neuron(self._num_inputs))



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
    file = pd.read_csv('test_data.csv').values
    print(file[0])
    inputs = len(file[0])
    num_neurons = 10  # arbitrary number at this point

    Net(num_inputs=inputs, num_neurons=num_neurons)


if __name__ == '__main__':
    main()
