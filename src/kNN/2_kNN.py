# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:17:56 2016

@author: Thom
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder


class KNN():
    k = 3
    X_data = []
    X_target = []

    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        # self.X_target = 0
        # self.X_data = 0

    def fit(self, X_data, X_target):
        # self.X_data, self.X_target = X_data, X_target
        self.X_data = X_data
        self.X_target = X_target
        return

    def predict(self, y_data):

        nInputs = np.shape(y_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            # Compute distances
            distances = np.sum((self.X_data-y_data[n,:])**2,axis=1)

            # Identify the nearest neighbors
            indices = np.argsort(distances,axis=0)

            classes = np.unique(self.X_target[indices[:self.k]])
            if len(classes)==1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes)+1)
                for i in range(self.k):
                    counts[self.X_target[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest

    def score(self, y_data, y_target):
        prediction = self.predict(y_data)

        return np.sum(prediction == y_target) / len(y_target)

def loadDataset(dataset):
    if dataset == 'iris':
        df = pd.io.parsers.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        )
        # df.column = ["sl", "sw", "pl", "pw", "class"]
        df[4] = preprocessing.LabelEncoder().fit_transform(df[4])
        # iris = df.as_matrix(columns=[df.columns])
        df = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(df).toarray())
        # print(df)
    elif dataset == 'cars':
        df = pd.io.parsers.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
        header=None,
        )
        # df.column = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        for i in df:
            df[i] = preprocessing.LabelEncoder().fit_transform(df[i])
        df = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(df).toarray())
        # print(df)
    elif dataset == 'breasts':
        df = pd.io.parsers.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        header=None,
        )
        df = df.replace({'?': float('nan')})
        df = pd.DataFrame(
            OneHotEncoder(dtype=np.int)._fit_transform(
                Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(df.replace({'?': np.nan}))
            ).toarray()
        )
    return df.values

# Bro. Burton's Magic Code with Thom Tweaks
def split_data(dataset, split_amount):
    data, target= dataset[:,:-1], dataset[:,-1]

    split_index = int(split_amount * len(data))

    indices = np.random.permutation(len(data))

    X_data = data[indices[:split_index]]
    X_target = target[indices[:split_index]]

    y_data = data[indices[split_index:]]
    test_target = target[indices[split_index:]]
    return (X_data, X_target, y_data, test_target)

def process_data(X_data, X_target, y_data, y_target):
    sknn = KNeighborsClassifier(n_neighbors=3)
    sknn.fit(X_data, X_target)
    # prediction = sknn.predict(y_data)
    # print("Prediction: %s%%" % prediction)

    score = sknn.score(y_data, y_target)
    print(" SkLearn Score: %s%%" % round(score * 100, 2))


    return

def main():
    split = 0.7

    # Irisis Data
    print("Irises Data:")
    iris = loadDataset('iris')

    split_data(iris, split)
    X_data, X_target, y_data, y_target = split_data(iris, split)

    process_data(X_data, X_target, y_data, y_target)

    knn = KNN(k_neighbors = 3)
    knn.fit(X_data, X_target)
    score = knn.score(y_data, y_target)

    print(" My Imp Score: %s%%" % round(score * 100, 2))
    print()

    # Cars Data
    print("Cars Data:")
    cars = loadDataset('cars')

    X_data, X_target, y_data, y_target = split_data(cars, split)

    process_data(X_data, X_target, y_data, y_target)

    knn = KNN(k_neighbors = 3)
    knn.fit(X_data, X_target)
    score = knn.score(y_data, y_target)

    print(" My Imp Score: %s%%" % round(score * 100, 2))
    print()

    #  Breast Cancer Data
    print("Breast Cancer Data:")
    breasts = loadDataset('breasts')
    X_data, X_target, y_data, y_target = split_data(breasts, split)

    process_data(X_data, X_target, y_data, y_target)

    knn = KNN(k_neighbors = 3)
    knn.fit(X_data, X_target)
    score = knn.score(y_data, y_target)

    print(" My Imp Score: %s%%" % round(score * 100, 2))

if __name__ == '__main__':
    main()


