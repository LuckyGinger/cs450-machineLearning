from collections import Counter

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from learndata import LearnData as ld
from learndata import ManipulateData as md



def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


def calc_info_gain(data, targets, feature):
    gain = 0
    n_data = len(data)
    # List the values that feature can take
    values = []
    for datapoint in data:
        if datapoint[feature] not in values:
            values.append(datapoint[feature])

    feature_counts = np.zeros(len(values))
    # print(len(feature_counts))
    entropy = np.zeros(len(values))
    the_entropy = 0.0
    value_index = 0
    # print(len(values))
    # Find where those values appear in data[feature] and the corresponding class
    for value in values:
        data_index = 0
        new_targets = []
        for datapoint in data:
            if datapoint[feature] == value:
                # feature_counts[value_index] += 1
                new_targets.append(targets[data_index])
            data_index += 1
        # print(feature_counts)


        # Get the values in new_targets
        class_values = []
        for aclass in new_targets:
            if class_values.count(aclass) == 0:
                class_values.append(aclass)
        class_counts = np.zeros(len(class_values))
        class_index = 0
        for classValue in class_values:
            for aclass in new_targets:
                if aclass == classValue:
                    class_counts[class_index] += 1
            class_index += 1
        # print(class_values)
        # print(class_counts)
        # print(class_index)
        temp_count = value_index - 1
        for class_index in range(len(class_values)):
            # temp_count += 1
            entropy[value_index] += calc_entropy(float(class_counts[class_index]) / sum(class_counts))

            # print(entropy)
            # gain += float(feature_counts[value_index]) / n_data * entropy[value_index]

        the_entropy += (sum(class_counts)/len(data)) * (entropy[value_index])
        value_index += 1

    # print(entropy)
    # print(value_index)

    return the_entropy

def make_tree(data, targets):

    # Various initializations suppressed
    n_data = len(data)
    n_features = len(data[0])


    default = Counter(targets).most_common(1)[0][0]
    # if n_data == 0 or n_features == 0:
    #     # Have reached an empty branch
    #     return default
    # elif targets.count(targets[0]) == n_data:
    #     # Only 1 class remains
    #     return targets[0]
    # else:
    #     Choose which feature is best

    # branch = np.zeros(n_features)
    # print(branch)
    for feature in range(n_features):
        print(calc_info_gain(data, targets, feature))
        # branch[feature] = calc_info_gain(data, targets, feature)

    # print(branch)


        # print(g)

    # print(total)
            # gain[feature] = totalEntropy - g
        # best_feature = np.argmax(gain)
    #     tree = {feature_names[best_feature]: {}}
    #
    #     values = []
    #     for datapoint in data:
    #         if datapoint[feature] not in values:
    #             values.append(datapoint[feature])
    #
    #     # Find the possible feature values
    #     for value in values:
    #         new_data = []
    #         new_targets = []
    #         new_names = ""
    #         index = 0
    #
    #         # Find the datapoints with each feature value
    #         for datapoint in data:
    #             if datapoint[best_feature] == value:
    #                 if best_feature == 0:
    #                     datapoint = datapoint[1:]
    #                     new_names = feature_names[1:]
    #                 elif best_feature == n_features:
    #                     datapoint = datapoint[:-1]
    #                     new_names = feature_names[:-1]
    #                 else:
    #                     datapoint = datapoint[:best_feature]
    #                     datapoint.extend(datapoint[best_feature + 1:])
    #                     new_names = feature_names[:best_feature]
    #                     new_names.extend(feature_names[best_feature + 1:])
    #                 new_data.append(datapoint)
    #                 new_targets.append(targets[index])
    #             index += 1
    #         # Now recurse to the next level
    #         subtree = make_tree(new_data, new_targets, new_names)
    #         # and on returning, add the subtree on to the tree
    #         tree[feature_names[best_feature]][value] = subtree
    #     return tree
    return

def main():
    split = 0.7

    # Irisis Data
    print("Irises Data:")
    iris = ld.load_dataset('iris')
    X_data, X_target, y_data, y_target = md.split_data(iris, split)
    # print(X_data)
    # print(X_target)
    make_tree(X_data, X_target)

if __name__ == '__main__':
    main()
