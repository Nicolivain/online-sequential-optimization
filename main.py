"""
Sorbonne University
Master M2A
Convex sequential Optimization
Ollivain Nicolas
Liautaud Paul
Le Boudec Lise

Main file 
""" 

import numpy as np
import pandas as pd
import pathlib as Path

from Algorithms.GD import GradientDescent
from utils import LinearSVM

############################### Read and prepare data ###############################

mnist_train=pd.read_csv('../mnist_train.csv', sep=',',header=None) # reading
train_data = mnist_train.values[:, 1:] # Extract data
train_data = train_data / np.max(train_data) # normalize data
train_labels = mnist_train.values[:, 0] # Extract labels
train_labels[np.where(train_labels != 0)] = -1 # if labels is not => -1 else 0 

mnist_test=pd.read_csv('../mnist_test.csv', sep=',',header=None)
test_data = mnist_test.values[:, 1:]
test_data = test_data / np.max(test_data)
test_labels = mnist_test.values[:, 0]
test_labels[np.where(test_labels != 0)] = -1

############################### Test algorithms ###############################
lr = 0.01
nepoch = 10
lbd = 1
verbose = 5

model = LinearSVM()

GDparams, GDloss = GradientDescent(model, train_data, train_labels,lr, nepoch, lbd, verbose)
pred_test_labels = model.predict(GDparams, test_data)
GDacc = model.accuracy(test_labels, pred_test_labels)
print(f'After {nepoch} epoch, GD algorithm has a loss of {GDloss[-1]} and accuracy {GDacc}')