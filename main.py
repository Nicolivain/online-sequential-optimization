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

from Algorithms.Adam import adaMax, adaMaxTemporal, adam, adamP, adamTemporal, adamproj
from Algorithms.Explo import sbeg, sreg
from Algorithms.GD import GradientDescent, projected_gd
from Algorithms.SGD import sgd, projected_sgd
from Algorithms.RFTL import adagrad, seg, smd
from Algorithms.ONS import ons
from Models.LinearSVM import LinearSVM
from utils import *


# --- PARAMETERS ---



np.random.seed(123)

lr = 0.1
nepoch = 50
lbd = 1/3
z = 100
gamma = 1/8
verbose = 1

alg_to_run = ['gd', 'c_gd', 'sgd', 'c_sgd', 'smd', 'seg', 'adagrad', 'ons',
              'sreg', 'sbeg', 'adam', 'adamproj', 'adamp', 'adamax', 'adamtemp', 'adamaxtemp']
# alg_to_run = ['gd', 'c_gd']

############################### Read and prepare data ###############################

mnist_train = pd.read_csv('mnist_train.csv', sep=',', header=None)   # Reading
# Extract data
train_data = mnist_train.values[:, 1:]
# Normalize data
train_data = train_data / np.max(train_data)
train_data = np.c_[train_data, np.ones(
    train_data.shape[0])]         # Add intersept
# Extract labels
train_labels = mnist_train.values[:, 0]
# if labels is not 0 => -1 (Convention chosen)
train_labels[np.where(train_labels != 0)] = -1
# if label is 0 ==> 1
train_labels[np.where(train_labels == 0)] = 1

mnist_test = pd.read_csv('mnist_test.csv', sep=',', header=None)
test_data = mnist_test.values[:, 1:]
test_data = test_data / np.max(test_data)
test_data = np.c_[test_data, np.ones(test_data.shape[0])]
test_labels = mnist_test.values[:, 0]
test_labels[np.where(test_labels != 0)] = -1
test_labels[np.where(test_labels == 0)] = 1

n, m = train_data.shape
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

############################### Test algorithms ###############################

# Unconstrained GD

if 'gd' in alg_to_run:
    model = LinearSVM(m)
    GDloss, wts = GradientDescent(
        model, train_data, train_labels, lr, nepoch, lbd, verbose)
    pred_test_labels = model.predict(test_data)
    GDacc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained GD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, GDloss[-1], GDacc))
    ax[0].plot(np.arange(nepoch), GDloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

# Constrained GD: projection on B1(z)

if 'c_gd' in alg_to_run:
    model = LinearSVM(m)
    GDprojloss, wts = projected_gd(
        model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    GDacc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained GD (radius {:2d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, z, GDprojloss[-1], GDacc))
    ax[0].plot(np.arange(nepoch), GDprojloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

# Unconstrained SGD

if 'sgd' in alg_to_run:
    model = LinearSVM(m)
    SGDloss, wts = sgd(model, train_data, train_labels,
                       lr, nepoch, lbd, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained SGD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SGDloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SGDloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

# Projected SGD

if 'c_sgd' in alg_to_run:
    model = LinearSVM(m)
    SGDprojloss, wts = projected_sgd(
        model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SGD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SGDprojloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SGDprojloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'smd' in alg_to_run:
    model = LinearSVM(m)
    SMDprojloss, wts = smd(model, train_data, train_labels,
                           lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SMD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SMDprojloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SMDprojloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'seg' in alg_to_run:
    model = LinearSVM(m)
    SEGloss, wts = seg(model, train_data, train_labels,
                       lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SEGloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adagrad' in alg_to_run:
    model = LinearSVM(m)
    Adagradloss, wts = adagrad(
        model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained Adagrad algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, Adagradloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adagradloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)

if 'ons' in alg_to_run: #try with rate instead of compute_accuracies
    model = LinearSVM(m)
    Onsloss, wts = ons(model, train_data, train_labels, nepoch, lbd, gamma, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, ONS algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, Onsloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Onsloss)
    accuracies = rate(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'sreg' in alg_to_run:
    model = LinearSVM(m)
    SREGloss, wts = sreg(model, train_data, train_labels,
                         lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SREG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SREGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SREGloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'sbeg' in alg_to_run:
    model = LinearSVM(m)
    SBEGloss, wts = sbeg(model, train_data, train_labels,
                         lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SBEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SBEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SBEGloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adam' in alg_to_run:
    model = LinearSVM(m)
    Adamloss, wts = adam(model, train_data, train_labels,
                         lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, Adamloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adamloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)


if 'adamproj' in alg_to_run:
    model = LinearSVM(m)
    AdamProjloss, wts = adamproj(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, projected adam algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdamProjloss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamProjloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adamp' in alg_to_run:
    p = 3
    model = LinearSVM(m)
    AdamPloss, wts = adamP(model, train_data, train_labels,
                           lr, nepoch, lbd, z, [0.9, 0.999], p, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam with norm L{:3d} algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, p, AdamPloss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamPloss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adamtemp' in alg_to_run:
    model = LinearSVM(m)
    AdamTemploss, wts = adamTemporal(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdamTemploss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamTemploss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adamax' in alg_to_run:
    model = LinearSVM(m)
    AdaMaxLoss, wts = adaMax(model, train_data, train_labels, lr, nepoch, lbd, z, [
                             0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxLoss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

if 'adamaxtemp' in alg_to_run:
    model = LinearSVM(m)
    AdaMaxTempLoss, wts = adaMaxTemporal(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxTempLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxTempLoss)
    accuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(accuracies)
    errors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(errors)

# Log scale
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_yscale('log')

# legend
ax[0].legend(alg_to_run)
ax[1].legend(alg_to_run)
ax[2].legend(alg_to_run)
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[2].set_title('Error')

plt.savefig('LossAccuraciesErrors.jpg')
plt.show()
