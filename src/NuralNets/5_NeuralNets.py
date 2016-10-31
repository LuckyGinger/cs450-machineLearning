import random
import bunch
from sklearn import preprocessing
import numpy as np
import pandas as pd
from learndata import LearnData as ld
from learndata import ManipulateData as md
from collections import Counter


class Net:
    def __init__(self, data, target, num_neurons=[1], bias=-1, learn_rate=0.2):
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
        self.values = self.calc_targets()

        self.double_diff_out = []

        # print(len(Counter(target).most_common()))
        # print(Counter(target).most_common())



        self._create_neurons()
        print("Before: ")
        # self.display_net()
        print()
        for i in range(100000):
            # print("Forward: ")
            # self.display_net()
            self.calculate_outputs()

            # print("Forward Display: ")
            # self.display_net()
            # print()
            # print("Backwards: ")
            # self.backwards()
            # print("Backwards Display: ")
            # print()
            if (i % 1000 == 0):
                print(self.outputs)
                print(self.double_diff_out)
                print(i, "Accuracy: " + str(self.accuracy()) + "%")
            # indices = np.random.permutation(len(data))
            # self.data = [np.insert(i, 0, self.bias) for i in data[indices[:]]]
            # print(self.data)
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

    def calc_targets(self):
        targets = [i[0] for i in Counter(self.targets).most_common()]
        values = {}

        for i, target in enumerate(targets):
            temp_val = list("0" * self._num_outputs)
            temp_val[i] = "1"
            values["".join(temp_val)] = target
        return values

    def accuracy(self):
        outputs = []
        for out in self.outputs:
            outputs.append("".join(str(i) for i in out))

        # Get list of computed targets
        out = [self.values[i] for i in outputs]

        #  Get list of matches
        total = 0
        if not len(out) == len(self.targets):
            print("Output error")
            print("out: " + str(len(out)), out)
            print("self.targets: " + str(len(self.targets)), self.targets)
            exit()

        for i, j in zip(out, self.targets):  # assuming the lists are of the same length
            if i == j:
                total += 1
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

    def _weight_updater(self, row):
        for index_layer, layer in enumerate(self.neurons):
            for index_neuron, neuron in enumerate(layer):
                for index_weight, weight in enumerate(neuron.get_weights()):
                    if index_weight == 0:
                        new_weight = (weight - self.learn_rate * neuron.get_error() * self.bias)
                    elif index_layer == 0:
                        new_weight = (weight - self.learn_rate * neuron.get_error() * row[index_weight])
                    else:
                        a = self.neurons[index_layer - 1][index_weight - 1].get_error()
                        new_weight = (weight - self.learn_rate * neuron.get_a() * a)
                    neuron.set_weight(index_weight, new_weight)

    def backwards(self, row, outputs):
        bias_error = []
        for index_layer, layer in enumerate(reversed(self.neurons)):
            for index_neuron, neuron in enumerate(layer):
                a = neuron.get_a()
                if index_layer > 0:  # Calculate Error for hidden layers
                    sum_weights = 0
                    # if index_neuron == 0: #  Need to handle the bias
                    #     for prev_neuron in self.neurons[len(self.neurons) - index_layer]:
                    #         sum_weights += prev_neuron.get_weight(index_neuron) * prev_neuron.get_error()
                    #     bias_error.append(self.bias * (1 - self.bias) * sum_weights)
                    for prev_neuron in self.neurons[len(self.neurons) - index_layer]:
                        # print(index_neuron, prev_neuron)
                        sum_weights += prev_neuron.get_weight(index_neuron + 1) * prev_neuron.get_error()
                    neuron.set_error(a * (1 - a) * sum_weights)
                else:  # Calculate Error for output layers
                    error = a * (1 - a) * (a - outputs[index_neuron])
                    neuron.set_error(error)

        self._weight_updater(row)

    def calculate_outputs(self):

        def sigmoid(x):
            # TODO: Figure out the "OverflowError: math range error"
            return 1 / (1 + np.math.exp(-x))

        self.outputs = []
        self.double_diff_out = []

        temp_h = 0
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
            diff_out = []
            for i, x in enumerate(self.neurons[-1]):
                x = x.get_a()
                # out.append(x)
                if x > highest:
                    highest = x
                    index = i

            for i in range(len(self.neurons[-1])):
                if i == index:
                    # out += "1"
                    out.append(1)
                    diff_out.append(self.neurons[-1][i].get_a())
                else:
                    # out += "0"
                    out.append(0)
                    diff_out.append(self.neurons[-1][i].get_a())
            self.double_diff_out.append(diff_out)
            self.outputs.append(out)
            # print(out)
            # print(diff_out)
            #  Do Back prop stuffs here
            # self.display_net()
            self.backwards(data, out)
            # print(out)
            # print(diff_out)
            # self.display_net()

            # print("Row " + str(index_data) + ": ")
            # self.display_net()
            # print(out)
            # print()
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
    # # # for i in X_data:
    # # #     print(i)
    # num_neurons = [1]  # array for the number of neurons in each hidden layer
    # Net(X_data, X_target, num_neurons=num_neurons)
    # # print()


    # print("Pima Indians Data:")
    # pima = ld.load_dataset('pima')
    # X_data, X_target, y_data, y_target = md.split_data(pima, split)
    # # for i in X_data:
    # #     print(i)
    # Net(X_data, X_target, num_neurons=num_neurons)
    # print()


if __name__ == '__main__':
    main()
