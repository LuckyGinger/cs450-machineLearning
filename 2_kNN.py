# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:17:56 2016

@author: Thom
"""

import sys
import argparse
import random
from sklearn import datasets, preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import  scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier

# KNN Class (Step 5)
class KNN():
    def __init__( self ):
        self.knn = 0

    def train(self, train, target):

        # TODO: Make it learn (~*.*)~
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(train, target)
        g = self.knn.kneighbors_graph(train)
        stuff = self.knn.get_params(deep=True)
        print("Params: \n%s" % stuff)

    def predict(self, array):
        predictions = self.knn.predict(array)
        print("Prediction: \n%s" % predictions)

    def display_graph(self, array, targets):
        g = self.knn.kneighbors_graph(array)
        print("Graph: \n%s" % g.toarray())
        s = self.knn.score(array, targets)
        print("Score: \n%s" % s)

def train(train, target):
    return

# Bro. Burton's Magic Code with Thom Tweaks
def split_data(dataset, split_amount):
    target = np.array([i[-1] for i in dataset])
    data = np.array([np.delete(i, -1) for i in dataset])

    split_index = split_amount * len(data)

    indices = np.random.permutation(len(data))

    train_data = data[indices[:split_index]]
    train_target = target[indices[:split_index]]

    test_data = data[indices[split_index:]]
    test_target = target[indices[split_index:]]

    return (train_data, train_target, test_data, test_target)

def main():
    # load dataset (Steps 1 & 2)
    iris = datasets.load_iris()

    df = pd.io.parsers.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    )
    df[4] = preprocessing.LabelEncoder().fit_transform(df[4])
    array = df.as_matrix(columns=[df.columns])
    print(array)
    # df = pd.io.parsers.read_csv(
    # 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    # header=None,
    # )
    # df = df.replace(to_replace='?', value=np.nan)
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # replace ? with the mean of all numbers in column
    # imp.fit(df)
    # df = imp.transform(df)
    # df[4] = preprocessing.LabelEncoder().fit_transform(df[4])
    # df = df.replace({'?': float('nan')})

    # mask = np.isnan(df)
    # masked_arr = np.ma.masked_array(arr, mask)
    # means = np.mean(masked_arr, axis=0)
    # print(masked_arr.filled(means))
    # mean = df.mean()
    # df = df.fillna(mean)
    # array = df.where(pd.notnull(df), df.mean(), axis='columns')
    # print(df)
    # array = df.as_matrix(columns=[df.columns])
    # array = [[1000 if j == 'nan' else j for j in i] for i in array]
    # array = np.ma.masked_array(array, np.isnan(array))
    # [print(j) for j in array]
    # print(np.mean(array, axis=0))
    # print(array)


    # randomize dataset instances (Step 3)
    split = 0.7
    x_data, x_target, y_data, y_target = split_data(array, split)
    print(x_data)
    print(x_target)
    print(y_data)
    print(y_target)


    # instantiate classifier (Step 6)
    # classifier = KNN()
    # # Make it learn (~*.*)~
    # classifier.train(train["data"], train["target"])
    # # Test its knowledge
    # prediction = classifier.predict(test["data"])
    # classifier.display_graph(test["data"], test["target"])


    # determine accuracy (Step 7)
    # count = 0
    # for prediction, target in zip(prediction, test["target"]):
    #     if prediction == target:
    #         count += 1

    # acc = round((count / len(test["target"]) * 100), 2)
    # print("Accuracy: %s%%" % acc)


    #print(values)
    #print(train["data"][0])
    #print(train["data"][0][0])

    # Show the data (the attributes of each instance)
    #print(iris.data)

    # Show the target values (in numeric format) of each instance
    #print(iris.target)

    # Show the actual target names that correspond to each number
    #print(iris.target_names)


if __name__ == '__main__':
    main()


