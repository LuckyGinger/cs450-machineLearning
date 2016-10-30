import random
import bunch
from sklearn import preprocessing
import numpy as np
import pandas as pd
from learndata import LearnData as ld
from learndata import ManipulateData as md
from collections import Counter


class Net:
    def __init__(self, data, target, num_neurons=[1], bias=-1, learn_rate=0.1):
        self._num_inputs = len(data[0])  # const value
        self._num_neurons = num_neurons  # const value
        self._num_outputs = len(Counter(target).most_common())
        self.learn_rate = learn_rate
        self.bias = bias
        self.data = [np.insert(i, 0, self.bias) for i in data]
        # print(self.data)
        self.targets = target
        self.inputs = []
        self.neurons = []
        self.outputs = []

        # print(len(Counter(target).most_common()))
        # print(Counter(target).most_common())

        self._create_neurons()
        print("Before: ")
        self.display_net()
        print()
        for i in range(1):
            print("Forward: ")
            self.calculate_outputs()
            print("Forward Display: ")
            self.display_net()
            print()
            print("Backwards: ")
            self.backwards()
            print("Backwards Display: ")
            self.display_net()
            print()
            print(i, "Accuracy: " + str(self.accuracy()) + "%")
        # print(self.outputs)
        # print("before")
        # print("Accuracy: " + str(self.accuracy()) + "%")
        # self.display_net()
        # self.backwards()
        # print("after")
        # print("Accuracy: " + str(self.accuracy()) + "%")
        # self.display_net()
        # print(self.outputs)
        # print(len(self.outputs))
        # print(self.outputs)

        # print(self.outputs)
        print(self.accuracy())
        # print("Accuracy: " + str(self.accuracy()) + "%")
        # for i in range(self._num_neurons):
        #     print(self.neurons[i])

        return

    def accuracy(self):

        # TODO: update accuracy to adjust outputs array to an array of strings

        outputs = []
        for out in self.outputs:
            outputs.append("".join(str(i) for i in out))

        targets = [i[0] for i in Counter(self.targets).most_common()]
        values = {}

        #  Create dictionary of targets
        for i, target in enumerate(targets):
            temp_val = list("0" * self._num_outputs)
            temp_val[i] = "1"
            values["".join(temp_val)] = target

        # Get list of computed targets
        out = [values[i] for i in outputs]
        #  Get list of matches
        total = 0
        for i, j in zip(out, self.targets):  # assuming the lists are of the same length
            if i == j:
                total += 1
        # print(out)
        # print(self.targets)
        # total = set(out) & set(self.targets)
        # print(total)
        #  Return the accuracy
        return (total / len(self.targets) * 100)

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

    def _weight_updater(self):
        return

    # for data in self.data:
    #     data = np.insert(data, 0, self.bias)
    #     for index_layer, layer in enumerate(self.neurons):
    #         for index_neuron, neuron in enumerate(layer):
    #             for index_weight, weight in enumerate(neuron.get_weights()):

    def backwards(self):
        #  Start with the Output Neurons
        # temp_data = [np.insert(i, 0, self.bias) for i in self.data

        for index_data, data in enumerate(self.data):
            # data = np.insert(data, 0, self.bias)
            for index_layer, layer in enumerate(reversed(self.neurons)):
                for index_neuron, neuron in enumerate(layer):
                    a = neuron.get_a()
                    if index_layer > 0:  # Calculate Error for hidden layers
                        sum_weights = 0
                        for prev_neuron in self.neurons[len(self.neurons) - index_layer]:
                            # print(index_neuron, prev_neuron)
                            sum_weights += prev_neuron.get_weight(index_neuron) * prev_neuron.get_error()
                        neuron.set_error(a * (1 - a) * sum_weights)
                    else:  # Calculate Error for output layers
                        neuron.set_error(a * (1 - a) * (a - self.outputs[index_data][index_neuron]))
                    for index_weight, weight in enumerate(neuron.get_weights()):  # Adjust the weights
                        neuron.set_weight(index_weight, (weight - self.learn_rate * neuron.get_error() * a))
                # print(neuron)
            # self.display_net()
            print("Row " + str(index_data) + ": ")
            self.display_net()
            print()

    def calculate_outputs(self):

        def sigmoid(x):
            # TODO: Figure out the "OverflowError: math range error"
            return 1 / (1 + np.math.exp(-x))

        temp_h = 0
        #  Todo: OPTIMIZE
        for index_data, data in enumerate(self.data):
            # data = np.insert(data, 0, self.bias)
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
                    neuron.set_h(temp_h)
                    neuron.set_a(sigmoid(temp_h))
                    temp_h = 0
            highest = -1
            index = 0
            # out = np.zeros(len(self.neurons[-1]))
            # out = ""
            out = []
            for i, x in enumerate(self.neurons[-1]):
                x = x.get_h()
                # out.append(x)
                if x > highest:
                    highest = x
                    index = i

            for i in range(len(self.neurons[-1])):
                if i == index:
                    # out += "1"
                    out.append(1)
                else:
                    # out += "0"
                    out.append(0)
            self.outputs.append(out)
            print("Row " + str(index_data) + ": ")
            self.display_net()
            print(out)
            print()
        return


class Neuron:
    def __init__(self, num_inputs=1):
        self.weights = []
        self.h = 0
        self.a = 0
        self.error = 0
        self._create_neuron(num_inputs)
        return

    def __str__(self):
        return "h: " + str(self.h) + " a: " + str(self.a) + " error: " + str(self.error) + " " + str(self.get_weights())

    def append_weight(self, weight):
        self.weights.append(weight)

    def get_weights(self):
        return self.weights

    def get_weight(self, index):
        return self.weights[index]

    def set_weight(self, index, weight):
        self.weights[index] = weight

    def set_h(self, h):
        self.h = h

    def set_a(self, a):
        self.a = a

    def set_error(self, error):
        self.error = error

    def get_error(self):
        return self.error

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
    split = 1
    file = pd.read_csv('test_data.csv').values
    print(file[0])
    print(file)
    inputs = len(file[0])
    num_neurons = [1]  # array for the number of neurons in each hidden layer
    X_data, X_target, y_data, y_target = md.split_data(file, split)
    print(X_data, X_target)
    Net(X_data, X_target, num_neurons=num_neurons)
    # print()

    # print("Irises Data:")
    # iris = ld.load_dataset('iris')
    # X_data, X_target, y_data, y_target = md.split_data(iris, split)
    # # for i in X_data:
    # #     print(i)
    # Net(X_data, X_target, num_neurons=num_neurons)
    # print()


    # print("Pima Indians Data:")
    # pima = ld.load_dataset('pima')
    # X_data, X_target, y_data, y_target = md.split_data(pima, split)
    # # for i in X_data:
    # #     print(i)
    # Net(X_data, X_target, num_neurons=num_neurons)
    # print()


if __name__ == '__main__':
    main()
