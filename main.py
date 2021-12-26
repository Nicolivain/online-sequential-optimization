"""
Sorbonne University
Master M2A
Convex sequential Optimization
Olivain Nicolas
Liautaud Paul
Le Boudec Lise

Main file 
""" 

import numpy as np
import pandas as pd
import pathlib as Path
from Algorithms.Explo import sbeg, sreg

from Algorithms.GD import GradientDescent, projected_gd
from Algorithms.SGD import sgd, projected_sgd
from Algorithms.RFTL import adagrad, seg, smd
from Models.LinearSVM import LinearSVM
from utils import *


# --- PARAMETERS ---

lr          = 0.1
nepoch      = 100
lbd         = 1
z           = 10
verbose     = 1

alg_to_run = ['gd', 'c_gd', 'sgd', 'c_sgd', 'smd', 'seg', 'adagrad', 'sreg', 'sbeg']


############################### Read and prepare data ###############################

mnist_train=pd.read_csv('mnist_train.csv', sep=',', header=None)  # Reading
train_data = mnist_train.values[:, 1:]                               # Extract data
train_data = train_data / np.max(train_data)                         # Normalize data
train_data = np.c_[train_data, np.ones(train_data.shape[0])]         # Add intersept
train_labels = mnist_train.values[:, 0]                              # Extract labels
train_labels[np.where(train_labels != 0)] = -1                       # if labels is not 0 => -1 (Convention chosen)
train_labels[np.where(train_labels == 0)] = 1                        # if label is 0 ==> 1

mnist_test=pd.read_csv('mnist_test.csv', sep=',', header=None)
test_data = mnist_test.values[:, 1:]
test_data = test_data / np.max(test_data)
test_data = np.c_[test_data, np.ones(test_data.shape[0])]
test_labels = mnist_test.values[:, 0]
test_labels[np.where(test_labels != 0)] = -1
test_labels[np.where(test_labels == 0)] = 1

n, m = train_data.shape
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

############################### Test algorithms ###############################

# Unconstrained GD

if 'gd' in alg_to_run:
    model = LinearSVM(m)
    GDloss, wts = GradientDescent(model, train_data, train_labels, lr, nepoch, lbd, verbose)
    pred_test_labels = model.predict(test_data)
    GDacc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained GD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, GDloss[-1], GDacc))
    ax[0].plot(np.arange(nepoch), GDloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

# Constrained GD: projection on B1(z)

if 'c_gd' in alg_to_run:
    model = LinearSVM(m)
    GDprojloss, wts = projected_gd(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    GDacc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained GD (radius {:2d} algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, z, GDloss[-1], GDacc))
    ax[0].plot(np.arange(nepoch), GDprojloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

# Unconstrained SGD

if 'sgd' in alg_to_run:
    model = LinearSVM(m)
    SGDloss, wts = sgd(model, train_data, train_labels, lr, nepoch, lbd, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained SGD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SGDloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SGDloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

# Projected SGD

if 'c_sgd' in alg_to_run:
    model = LinearSVM(m)
    SGDprojloss, wts = projected_sgd(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SGD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SGDprojloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SGDprojloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'smd' in alg_to_run:
    model = LinearSVM(m)
    SMDprojloss, wts = smd(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SMD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SMDprojloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SMDprojloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'seg' in alg_to_run:
    model = LinearSVM(m)
    SEGloss, wts = seg(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SEGloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'adagrad' in alg_to_run:
    model = LinearSVM(m)
    Adagradloss, wts = adagrad(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained Adagrad algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, Adagradloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adagradloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'sreg' in alg_to_run:
    model = LinearSVM(m)
    SREGloss, wts = sreg(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SREG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SREGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SREGloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'sbeg' in alg_to_run:
    model = LinearSVM(m)
    SBEGloss, wts = sbeg(model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SBEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SBEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SBEGloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)

# Log scale
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')

# legend
ax[0].legend(alg_to_run)
ax[1].legend(alg_to_run)
plt.show()

