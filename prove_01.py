# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:41:30 2016

@author: Thom
"""
import random
from sklearn import datasets

# HardCoded Class (Step 5)
class HardCoded():
 #   def __init__(self, size):
    def train(self, dataset):
        # TODO: Make it learn (~*.*)~
        return
    
    def predict(self, array):
        # TODO: Make it predict ~(*.*~)
        predictions = []
        for item in array:
            predictions.append(0)
        return predictions
        
# load dataset (Steps 1 & 2)
iris = datasets.load_iris()

# randomize dataset instances (Step 3)
data = list(zip(iris.data, iris.target))
random.shuffle(data)
iris.data, iris.target = zip(*data)

# split up the data 70/30 (Step 4)
train = {"data": [], "target": []}
test = {"data": [], "target": []}
for i in range(0, len(iris.data)):
    if i < (len(iris.data) * .7): #split 70/30 
        train["data"].append(iris.data[i])
        train["target"].append(iris.target[i])
    else:
        test["data"].append(iris.data[i])
        test["target"].append(iris.target[i])

print("Training Set")
print(train)
print("Test Set")
print(test)

# instantiate classifier (Step 6)
classifier = HardCoded()
# Make it learn (~*.*)~
classifier.train(train)
# Test its knowledge
prediction = classifier.predict(test["data"])


# determine accuracy (Step 7)
count = 0
for prediction, target in zip(prediction, test["target"]):
    if prediction == target:
        count += 1
        
acc = round((count / len(test["target"]) * 100), 2)
print("Accuracy: %s%%" % acc)


#print(values)
#print(train["data"][0])
#print(train["data"][0][0])

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)



    