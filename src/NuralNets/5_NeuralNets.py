import random
import bunch
from sklearn import preprocessing
import numpy as np
import pandas as pd
from learndata import LearnData as ld
from learndata import ManipulateData as md
from collections import Counter


class Net:
    def __init__(self, data, target, num_neurons=[1], bias=-1):
        self._num_inputs = len(data[0])  # const value
        self._num_neurons = num_neurons  # const value
        self._num_outputs = len(Counter(target).most_common())
        self.data = data
        self.targets = target
        self.bias = bias
        self.inputs = []
        self.neurons = []
        self.outputs = []

        # print(len(Counter(target).most_common()))
        # print(Counter(target).most_common())

        self._create_neurons()
        self.calculate_outputs()

        # print(len(self.outputs))
        # print(self.outputs)

        # self.display_net()
        # print(self.outputs)
        print("Accuracy: " + str(self.accuracy()) + "%")
        # for i in range(self._num_neurons):
        #     print(self.neurons[i])

        return

    def accuracy(self):
        targets = [i[0] for i in Counter(self.targets).most_common()]
        values = {}

        #  Create dictionary of targets
        for i, target in enumerate(targets):
            temp_val = list("0" * self._num_outputs)
            temp_val[i] = "1"
            values["".join(temp_val)] = target

        #  Get list of computed targets
        out = [values[i] for i in self.outputs]
        #  Get list of matches
        total = out == self.targets
        #  Return the accuracy
        return round(Counter(total)[True]/len(self.targets)*100, 2)

    def display_net(self):
        for i, x in enumerate(self.neurons):
            print("Layer #" + str(i) + ":")
            for j, y in enumerate(x):
                print("\tNeuron #" + str(j) + ": " + str(y))

    def _create_neurons(self):
        #  Create Hidden Neurons
        for x, j in enumerate(self._num_neurons):
            self.neurons.append([])  # To create j number of layers
            for i in range(j):  # To create i number of neurons in j layers
                if x == 0:  # If it's the first layer of Neurons after the inputs
                    self.neurons[x].append(Neuron(self._num_inputs))
                else:
                    self.neurons[x].append(Neuron(self._num_neurons[x - 1]))

        # Create Output Neurons/Last layer
        self.neurons.append([])
        for i in range(self._num_outputs):
            self.neurons[-1].append(Neuron(self._num_neurons[-1]))

        return

    def calculate_outputs(self):

        def sigmoid(x):
            # TODO: Figure out the "OverflowError: math range error"
            return 1 / (1 + np.math.exp(-x))

        temp_h = 0
        #  Todo: OPTIMIZE
        for data in self.data:
            data = np.insert(data, 0, self.bias)
            for index_layer, layer in enumerate(self.neurons):
                for index_neuron, neuron in enumerate(layer):
                    for index_weight, weight in enumerate(neuron.get_weights()):
                        if index_layer == 0:
                            temp_h += data[index_weight] * weight
                        else:
                            if index_weight == 0:  # Add the bias node
                                temp_h += self.bias * weight
                            else:
                                #  Get each neuron's "a" value from the layer before and multiply it by the weight
                                temp_h += self.neurons[index_layer - 1][index_weight - 1].get_a() * weight
                    neuron.set_h(round(temp_h, 2))
                    neuron.set_a(round(sigmoid(temp_h), 2))
                    temp_h = 0
            highest = -1
            index = 0
            # out = np.zeros(len(self.neurons[-1]))
            out = ""
            for i, x in enumerate(self.neurons[-1]):
                x = x.get_h()
                if x > highest:
                    highest = x
                    index = i
            for i in range(len(self.neurons[-1])):
                if i == index:
                    out += "1"
                else:
                    out += "0"
            self.outputs.append(out)
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
        self.h = 0
        self.a = 0
        self._create_neuron(num_inputs)
        return

    def __str__(self):
        return "h: " + str(self.h) + " a: " + str(self.a) + " " + str(self.get_weights())

    def append_weight(self, weight):
        self.weights.append(weight)

    def get_weights(self):
        return self.weights

    def get_weight(self, index):
        return self.weights[index]

    def set_h(self, h):
        self.h = h

    def set_a(self, a):
        self.a = a

    def get_h(self):
        return self.h

    def get_a(self):
        return self.a

    def _create_neuron(self, num_inputs):
        self.append_weight(round(random.uniform(-1, 1), 2))  # bias weight
        for _ in range(num_inputs):
            self.append_weight(round(random.uniform(-1, 1), 2))  # set the neuron weight here
        pass


def main():
    split = 0.7
    file = pd.read_csv('test_data.csv').values
    # print(file[0])
    inputs = len(file[0])
    num_neurons = [10, 2]  # array for the number of neurons in each hidden layer
    X_data, X_target, y_data, y_target = md.split_data(file, split)
    # print(X_data)
    Net(X_data, X_target, num_neurons=num_neurons)
    print()

    print("Irises Data:")
    iris = ld.load_dataset('iris')
    X_data, X_target, y_data, y_target = md.split_data(iris, split)
    # for i in X_data:
    #     print(i)
    Net(X_data, X_target, num_neurons=num_neurons)
    print()


    print("Pima Indians Data:")
    pima = ld.load_dataset('pima')
    X_data, X_target, y_data, y_target = md.split_data(pima, split)
    # for i in X_data:
    #     print(i)
    Net(X_data, X_target, num_neurons=num_neurons)
    print()

if __name__ == '__main__':
    main()
