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
nepoch = 10000
lbd = 1/3
z = 100
gamma = 1/8
verbose = 1

alg_to_run = ['gd', 'c_gd', 'sgd', 'c_sgd', 'smd', 'seg', 'adagrad', 'ons',
              'sreg', 'sbeg', 'adam', 'adamproj', 'adamp', 'adamax', 'adamtemp', 'adamaxtemp']

alg_to_run = ['ons', 'c_sgd']

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
    GDaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(GDaccuracies)
    GDerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(GDerrors)

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
    GDprojaccuracies = compute_accuracies(wts, test_data, test_labels, average=False) #no average for gd
    ax[1].plot(GDprojaccuracies)
    GDprojerrors = compute_errors(wts, test_data, test_labels, average=False)
    ax[2].plot(GDprojerrors)

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
    SGDaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SGDaccuracies)
    SGDerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SGDerrors)

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
    SGDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SGDprojaccuracies)
    SGDprojerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SGDprojerrors)

if 'smd' in alg_to_run:
    model = LinearSVM(m)
    SMDprojloss, wts = smd(model, train_data, train_labels,
                           lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SMD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SMDprojloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SMDprojloss)
    SMDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SMDprojaccuracies)
    SMDprojerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SMDprojerrors)

if 'seg' in alg_to_run:
    model = LinearSVM(m)
    SEGloss, wts = seg(model, train_data, train_labels,
                       lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SEGloss)
    SEGaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SEGaccuracies)
    SEGerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SEGerrors)

if 'adagrad' in alg_to_run:
    model = LinearSVM(m)
    Adagradloss, wts = adagrad(
        model, train_data, train_labels, lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained Adagrad algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, Adagradloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adagradloss)
    Adagradaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(Adagradaccuracies)
    Adagraderrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(Adagraderrors)

if 'ons' in alg_to_run:
    model = LinearSVM(m)
    ONSloss, wts = ons(model, train_data, train_labels, nepoch, lbd, gamma, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, ONS algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, ONSloss[-1], acc))
    ax[0].plot(np.arange(nepoch), ONSloss)
    ONSaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(ONSaccuracies)
    ONSerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(ONSerrors)

if 'sreg' in alg_to_run:
    model = LinearSVM(m)
    SREGloss, wts = sreg(model, train_data, train_labels,
                         lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SREG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SREGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SREGloss)
    SREGaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SREGaccuracies)
    SREGerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SREGerrors)

if 'sbeg' in alg_to_run:
    model = LinearSVM(m)
    SBEGloss, wts = sbeg(model, train_data, train_labels,
                         lr, nepoch, lbd, z, verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, constrained SBEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SBEGloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SBEGloss)
    SBEGaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SBEGaccuracies)
    SBEGerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SBEGerrors)

if 'adam' in alg_to_run:
    model = LinearSVM(m)
    Adamloss, wts = adam(model, train_data, train_labels,
                         lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, Adamloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adamloss)
    Adamaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(Adamaccuracies)
    Adamerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(Adamerrors)


if 'adamproj' in alg_to_run:
    model = LinearSVM(m)
    AdamProjloss, wts = adamproj(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, projected adam algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdamProjloss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamProjloss)
    AdamProjaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdamProjaccuracies)
    AdamProjerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdamProjerrors)

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
    AdamPaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdamPaccuracies)
    AdamPerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdamPerrors)

if 'adamtemp' in alg_to_run:
    model = LinearSVM(m)
    AdamTemploss, wts = adamTemporal(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdamTemploss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamTemploss)
    AdamTempaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdamTempaccuracies)
    AdamTemperrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdamTemperrors)

if 'adamax' in alg_to_run:
    model = LinearSVM(m)
    AdaMaxLoss, wts = adaMax(model, train_data, train_labels, lr, nepoch, lbd, z, [
                             0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxLoss)
    AdaMaxaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdaMaxaccuracies)
    AdaMaxerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdaMaxerrors)

if 'adamaxtemp' in alg_to_run:
    model = LinearSVM(m)
    AdaMaxTempLoss, wts = adaMaxTemporal(
        model, train_data, train_labels, lr, nepoch, lbd, z, [0.9, 0.999], verbose)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxTempLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxTempLoss)
    AdaMaxTempaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdaMaxTempaccuracies)
    AdaMaxTemperrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdaMaxTemperrors)

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

# t = np.arange(nepoch)
# fig = plt.figure()
# plt.plot(t, GDloss,
#         t, GDprojloss,
#         t, SGDloss,
#         t, SGDprojloss,
#         t, SMDprojloss,
#         t, SEGloss,
#         t, Adagradloss,
#         t, ONSloss,
#         t, SREGloss,
#         t, SBEGloss,
#         t, Adamloss,
#         t, AdamProjloss,
#         t, AdamPloss,
#         t, AdamTemploss,
#         t, AdaMaxLoss,
#         t, AdaMaxTempLoss)
# plt.legend(alg_to_run)
# plt.savefig('Losses.jpg')
# plt.show()
#
# t = np.arange(nepoch + 1)
# fig = plt.figure()
# plt.plot(t, GDaccuracies,
#         t, GDprojaccuracies,
#         t, SGDaccuracies,
#         t, SGDprojaccuracies,
#         t, SMDprojaccuracies,
#         t, SEGaccuracies,
#         t, Adagradaccuracies,
#         t, ONSaccuracies,
#         t, SREGaccuracies,
#         t, SBEGaccuracies,
#         t, Adamaccuracies,
#         t, AdamProjaccuracies,
#         t, AdamPaccuracies,
#         t, AdamTempaccuracies,
#         t, AdaMaxaccuracies,
#         t, AdaMaxTempaccuracies)
# plt.legend(alg_to_run)
# plt.savefig('Accuracies.jpg')
# plt.show()
#
# fig = plt.figure()
# plt.plot(t, GDerrors,
#         t, GDprojerrors,
#         t, SGDerrors,
#         t, SGDprojerrors,
#         t, SMDprojerrors,
#         t, SEGerrors,
#         t, Adagraderrors,
#         t, ONSerrors,
#         t, SREGerrors,
#         t, SBEGerrors,
#         t, Adamerrors,
#         t, AdamProjerrors,
#         t, AdamPerrors,
#         t, AdamTemperrors,
#         t, AdaMaxerrors,
#         t, AdaMaxTemperrors)
# plt.legend(alg_to_run)
# plt.savefig('Errors.jpg')
# plt.show()