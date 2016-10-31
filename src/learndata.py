from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


class LearnData():
    def __init__(self):
        return

    def load_dataset(dataset):
        if dataset == 'iris':
            df = pd.io.parsers.read_csv(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                header=None,
            )
            df[4] = preprocessing.LabelEncoder().fit_transform(df[4])
            # df.columns = ['sl', 'sw', 'pl', 'pw', 'class']
            # iris = df.as_matrix(columns=[df.columns])
            # df = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(df).toarray())
            # for i in df:
            #     df[i] = preprocessing.LabelEncoder().fit_transform(df[i])
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
                ).toarray())
        elif dataset == 'lenses':
            df = pd.io.parsers.read_csv(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
                header=None,
                delim_whitespace=True,
            )
            # del df[0]
            # df.astype(int)
        elif dataset == 'house-votes':
            df = pd.io.parsers.read_csv(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
                header=None,
            )
        elif dataset == 'pima':
            df = pd.io.parsers.read_csv(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                header=None,
            )
            # df = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(df).toarray())
            for i in df:
                df[i] = preprocessing.LabelEncoder().fit_transform(df[i])
        return df.values


class ManipulateData():
    def __init__(self):
        return

        # Bro. Burton's Magic Code with Thom Tweaks
    def split_data(dataset, split_amount=1):
            data, target = dataset[:, :-1], dataset[:, -1]

            indices = np.random.permutation(len(data))

            if split_amount == 1:
                return data[indices[:]], target[indices[:]], 0, 0
            else:
                split_index = int(split_amount * len(data))

                X_data = data[indices[:split_index]]
                X_target = target[indices[:split_index]]

                y_data = data[indices[split_index:]]
                y_target = target[indices[split_index:]]
                return (X_data, X_target, y_data, y_target)

    def process_data(X_data, X_target, y_data, y_target):
        sknn = KNeighborsClassifier(n_neighbors=3)
        sknn.fit(X_data, X_target)
        # prediction = sknn.predict(y_data)
        # print("Prediction: %s%%" % prediction)

        score = sknn.score(y_data, y_target)
        print(" SkLearn Score: %s%%" % round(score * 100, 2))

        return
