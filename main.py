"""
Sorbonne University
Master M2A
Convex sequential Optimization
Olivain Nicolas
Liautaud Paul
Le Boudec Lise

Main file 
"""

import time
import numpy as np
import pandas as pd
import pathlib as Path
import seaborn as sns

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
nepoch = 1000
lbd = 1/3  # or change to 1/5 for sbeg and sreg to get better results
Z = [100]
Zsbeg = [1000]
gamma = 1/8
verbose = 100

alg_to_run = ['gd', 'c_gd', 'sgd', 'c_sgd', 'smd', 'seg', 'adagrad', 'ons',
              'sreg', 'sbeg', 'adam', 'adam_fixlr', 'adamproj', 'adamp', 'adamax', 'adamtemp', 'adamaxtemp']

############################### Read and prepare data ###############################

mnist_train = pd.read_csv('mnist_train.csv', sep=',', header=None)   # Reading
# Extract data
train_data = mnist_train.values[:, 1:]
# Normalize data
train_data = train_data / np.max(train_data)
train_data = np.c_[train_data, np.ones(train_data.shape[0])]         # Add intercept
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

time_dict = {}

# Unconstrained GD

if 'gd' in alg_to_run:
    print("-----------GD----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    GDloss, wts = GradientDescent(
        model, train_data, train_labels, nepoch, lbd, verbose, lr)
    time_dict['gd'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    GDacc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained GD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, GDloss[-1], GDacc))
    ax[0].plot(np.arange(nepoch), GDloss, label='gd')
    GDaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(GDaccuracies, label='gd')
    GDerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(GDerrors, label='gd')

# Constrained GD: projection on B1(z)

if 'c_gd' in alg_to_run:
    for z in Z:  # play with the projection radius
        print("-----------c_GD - z="+str(z)+"----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        GDprojloss, wts = projected_gd(
            model, train_data, train_labels, nepoch, lbd, z, verbose, lr)
        time_dict['c_gd z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        GDacc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained GD (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, GDprojloss[-1], GDacc))
        ax[0].plot(np.arange(nepoch), GDprojloss, label='c_gd z='+str(z))
        GDprojaccuracies = compute_accuracies(
            wts, test_data, test_labels, average=False)  # no average for gd
        ax[1].plot(GDprojaccuracies, label='c_gd z='+str(z))
        GDprojerrors = compute_errors(
            wts, test_data, test_labels, average=False)
        ax[2].plot(GDprojerrors, label='c_gd z='+str(z))

# Unconstrained SGD

if 'sgd' in alg_to_run:
    print("-----------SGD----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    SGDloss, wts = sgd(model, train_data, train_labels,
                       nepoch, lbd, verbose, 1)
    time_dict['sgd'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, Unconstrained SGD algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, SGDloss[-1], acc))
    ax[0].plot(np.arange(nepoch), SGDloss, label='sgd')
    SGDaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(SGDaccuracies, label='sgd')
    SGDerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(SGDerrors, label='sgd')

# Projected SGD

if 'c_sgd' in alg_to_run:
    for z in Z:  # play with the projection radius
        print("-----------c_SGD - z=" + str(z)+"----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SGDprojloss, wts = projected_sgd(
            model, train_data, train_labels, nepoch, lbd, z, verbose, 1)
        time_dict['c_sgd z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SGD (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, SGDprojloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SGDprojloss, label='c_sgd z='+str(z))
        SGDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SGDprojaccuracies, label='c_sgd z='+str(z))
        SGDprojerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SGDprojerrors, label='c_sgd z='+str(z))

if 'smd' in alg_to_run:
    for z in Z:
        print("-----------SMD  - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SMDprojloss, wts = smd(model, train_data, train_labels,
                               lr, nepoch, lbd, z, verbose)
        time_dict['smd z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SMD (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, SMDprojloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SMDprojloss, label='smd z='+str(z))
        SMDprojaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SMDprojaccuracies, label='smd z='+str(z))
        SMDprojerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SMDprojerrors, label='smd z='+str(z))

if 'seg' in alg_to_run:
    for z in Z:
        print("-----------SEG - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SEGloss, wts = seg(model, train_data, train_labels, nepoch, lbd, z, lr, verbose)
        time_dict['seg z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SEG (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, SEGloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SEGloss, label='seg z='+str(z))
        SEGaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SEGaccuracies, label='seg z='+str(z))
        SEGerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SEGerrors, label='seg z='+str(z))

if 'adagrad' in alg_to_run:
    for z in Z:
        print("-----------Adagrad - z=" + str(z)+"----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        Adagradloss, wts = adagrad(model, train_data, train_labels, nepoch, lbd, z, verbose)
        time_dict['adagrad z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained Adagrad (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, Adagradloss[-1], acc))
        ax[0].plot(np.arange(nepoch), Adagradloss, label='adagrad z='+str(z))
        Adagradaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(Adagradaccuracies, label='adagrad z='+str(z))
        Adagraderrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(Adagraderrors,  label='adagrad z='+str(z))

if 'ons' in alg_to_run:
    for z in Z:
        print("-----------ONS - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        ONSloss, wts = ons(model, train_data, train_labels,
                           nepoch, lbd, gamma, z, verbose)
        time_dict['ons z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, ONS (radius {:3d} algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, ONSloss[-1], acc))
        ax[0].plot(np.arange(nepoch), ONSloss,  label='ons z='+str(z))
        ONSaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(ONSaccuracies,  label='ons z='+str(z))
        ONSerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(ONSerrors,  label='ons z='+str(z))

if 'sreg' in alg_to_run:
    for z in Z:
        print("-----------SREG - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SREGloss, wts = sreg(model, train_data, train_labels, nepoch, lbd, z, lr=1, verbose=verbose)
        time_dict['sreg z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SREG (radius {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, z, SREGloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SREGloss, label='sreg z='+str(z))
        SREGaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SREGaccuracies, label='sreg z='+str(z))
        SREGerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SREGerrors, label='sreg z='+str(z))

if 'sbeg' in alg_to_run:
    for z in Zsbeg:
        print("-----------SBEG - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        SBEGloss, wts = sbeg(model, train_data, train_labels, nepoch, lbd, z, lr=1, verbose=verbose)
        time_dict['sbeg z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, constrained SBEG algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, SBEGloss[-1], acc))
        ax[0].plot(np.arange(nepoch), SBEGloss, label='sbeg z='+str(z))
        SBEGaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(SBEGaccuracies, label='sbeg z='+str(z))
        SBEGerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(SBEGerrors, label='sbeg z='+str(z))

lr = 0.003
betas = [0.9, 0.999]

if 'adam' in alg_to_run:
    print("-----------Adam ----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    Adamloss, wts = adam(model, train_data, train_labels, lr, nepoch, lbd, betas, verbose)
    time_dict['adam'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, Adamloss[-1], acc))
    ax[0].plot(np.arange(nepoch), Adamloss, label='adam')
    Adamaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(Adamaccuracies, label='adam')
    Adamerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(Adamerrors, label='adam')

if 'adam_fixlr' in alg_to_run:
    print("-----------Adam fixed lr----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    AdamLRloss, wts = adam(model, train_data, train_labels, lr, nepoch, lbd, betas, verbose, adaptative_lr=False)
    time_dict['adam_fixlr'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam with fixed lr algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(nepoch, AdamLRloss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamLRloss, label='adam_fixlr')
    AdamLRaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdamLRaccuracies, label='adam_fixlr')
    AdamLRerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdamLRerrors, label='adam_fixlr')


if 'adamproj' in alg_to_run:
    for z in Z:
        print("-----------Adam projected - z=" + str(z) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        AdamProjloss, wts = adamproj(model, train_data, train_labels, lr, nepoch, lbd, z, betas, verbose)
        time_dict['adamproj z='+str(z)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, projected adam (radius = {:3d}) algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, z, AdamProjloss[-1], acc))
        ax[0].plot(np.arange(nepoch), AdamProjloss,
                   label='adamproj - z=' + str(z))
        AdamProjaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(AdamProjaccuracies, label='adamproj - z=' + str(z))
        AdamProjerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(AdamProjerrors, label='adamproj - z=' + str(z))

if 'adamp' in alg_to_run:
    P = [1, 2, 3]
    for p in P:
        print("-----------Adam norm : L" + str(p) + "----------- \n")
        model = LinearSVM(m)
        tic = time.time()
        AdamPloss, wts = adamP(model, train_data, train_labels, lr, nepoch, lbd, betas, p, verbose)
        time_dict['adamp p='+str(p)] = (time.time() - tic)
        pred_test_labels = model.predict(test_data)
        acc = accuracy(test_labels, pred_test_labels)
        print('After {:3d} epoch, adam with norm L{:3d} algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
            nepoch, p, AdamPloss[-1], acc))
        ax[0].plot(np.arange(nepoch), AdamPloss,
                   label='adam with norm L' + str(p))
        AdamPaccuracies = compute_accuracies(wts, test_data, test_labels)
        ax[1].plot(AdamPaccuracies, label='adam with norm L' + str(p))
        AdamPerrors = compute_errors(wts, test_data, test_labels)
        ax[2].plot(AdamPerrors, label='adam with norm L' + str(p))

if 'adamtemp' in alg_to_run:
    print("-----------Adam with temporal averaging ----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    AdamTemploss, wts = adamTemporal(model, train_data, train_labels, lr, nepoch, lbd, betas, verbose)
    time_dict['adamtemp'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, adam with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdamTemploss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdamTemploss, label='adamtemp')
    AdamTempaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdamTempaccuracies, label='adamtemp')
    AdamTemperrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdamTemperrors, label='adamtemp')

if 'adamax' in alg_to_run:
    print("-----------Adamax ----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    AdaMaxLoss, wts = adaMax(model, train_data, train_labels, lr, nepoch, lbd, betas, verbose)
    time_dict['adamax'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxLoss, label='adamax')
    AdaMaxaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdaMaxaccuracies, label='adamax')
    AdaMaxerrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdaMaxerrors, label='adamax')

if 'adamaxtemp' in alg_to_run:
    print("-----------Adamax with temporal averaging ----------- \n")
    model = LinearSVM(m)
    tic = time.time()
    AdaMaxTempLoss, wts = adaMaxTemporal(model, train_data, train_labels, lr, nepoch, lbd, betas, verbose)
    time_dict['adamaxtemp'] = (time.time() - tic)
    pred_test_labels = model.predict(test_data)
    acc = accuracy(test_labels, pred_test_labels)
    print('After {:3d} epoch, AdaMax with temporal averaging algorithm has a loss of {:1.6f} and accuracy {:1.6f}'.format(
        nepoch, AdaMaxTempLoss[-1], acc))
    ax[0].plot(np.arange(nepoch), AdaMaxTempLoss, label='adamaxtemp')
    AdaMaxTempaccuracies = compute_accuracies(wts, test_data, test_labels)
    ax[1].plot(AdaMaxTempaccuracies, label='adamaxtemp')
    AdaMaxTemperrors = compute_errors(wts, test_data, test_labels)
    ax[2].plot(AdaMaxTemperrors, label='adamaxtemp')


# Log scale
ax[0].set_xscale('log')
ax[0].set_yscale('logit')
ax[1].set_xscale('log')
ax[1].set_yscale('logit')
ax[2].set_xscale('log')
ax[2].set_yscale('logit')

# legend
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[2].set_title('Error')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[2].set_xlabel('Epochs')


plt.savefig('LossAccuraciesErrors.png')
plt.show()

plt.clf()
keys = list(time_dict.keys())
sns.barplot(x=keys, y=[time_dict[k] for k in keys])
plt.savefig('execution_time.png')
plt.show()


t = np.arange(nepoch)
fig = plt.figure()
plt.plot(t, GDloss,
         t, GDprojloss,
         t, SGDloss,
         t, SGDprojloss,
         t, SMDprojloss,
         t, SEGloss,
         t, Adagradloss,
         t, ONSloss,
         t, SREGloss,
         t, SBEGloss,
         t, Adamloss,
         t, AdamLRloss,
         t, AdamProjloss,
         t, AdamPloss,
         t, AdamTemploss,
         t, AdaMaxLoss,
         t, AdaMaxTempLoss)
plt.legend(alg_to_run)
plt.savefig('Losses.jpg')
plt.show()

t = np.arange(nepoch + 1)
fig = plt.figure()
plt.plot(t, GDaccuracies,
         t, GDprojaccuracies,
         t, SGDaccuracies,
         t, SGDprojaccuracies,
         t, SMDprojaccuracies,
         t, SEGaccuracies,
         t, Adagradaccuracies,
         t, ONSaccuracies,
         t, SREGaccuracies,
         t, SBEGaccuracies,
         t, Adamaccuracies,
         t, AdamLRaccuracies,
         t, AdamProjaccuracies,
         t, AdamPaccuracies,
         t, AdamTempaccuracies,
         t, AdaMaxaccuracies,
         t, AdaMaxTempaccuracies)
plt.legend(alg_to_run)
plt.xscale('log')
plt.yscale('log')
plt.savefig('Accuracies.jpg')
plt.show()

fig = plt.figure()
plt.plot(t, GDerrors,
         t, GDprojerrors,
         t, SGDerrors,
         t, SGDprojerrors,
         t, SMDprojerrors,
         t, SEGerrors,
         t, Adagraderrors,
         t, ONSerrors,
         t, SREGerrors,
         t, SBEGerrors,
         t, Adamerrors,
         t, AdamLRerrors,
         t, AdamProjerrors,
         t, AdamPerrors,
         t, AdamTemperrors,
         t, AdaMaxerrors,
         t, AdaMaxTemperrors)
plt.legend(alg_to_run)
plt.xscale('log')
plt.yscale('log')
plt.savefig('Errors.jpg')
plt.show()
