import errno
from Adult_utils import dataSet_summary, data_PreProcess, plot_learningCurve, check_balance, Dirichelet_sampling, dataset_slice
import tensorflow as tf
from tensorflow.keras import backend as K
import sklearn
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

import random
import pandas as pd
import numpy as np
import math
import os
import flwr as fl
import array
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE
import sys
from Adult_utils import *

"""
    True Negative rates pour le calcul de EOD
"""

def compute_TNR(x_test, y_test, model) :

    y_pred = model.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred.argmax(axis=1)).ravel()
    tnr = tn / (tn + fp)
    return round(tnr, 3)

"""
    Proportions de prédictions Positives pour la calcul de PSD
"""
def compute_PP(x_test, y_test, model) :

    y_pred = model.predict(x_test)
    #Seuil entre décision poitive et décision négative.
    positive_mask = tf.where(y_pred >= 0.5, 1, 0)
    positive_count = np.count_nonzero(positive_mask)
    total_count = y_pred.shape[0]
    positive_proportion = positive_count / total_count
    return round(positive_proportion, 3)


"""
    True Positive Rate pour le calcul de EOD
"""


#doit etre égale a Recall
def compute_TPR(x_test, y_test, model) :
    tp_metric = tf.keras.metrics.TruePositives()
    model.evaluate(x_test, y_test, verbose=0)
    tp_metric.update_state(y_test, model.predict(x_test))
    tp = tp_metric.result().numpy()
    actual_positive = tf.math.count_nonzero(y_test==1.0).numpy()
    tpr = tp / actual_positive
    return round(tpr, 3)


def EOD(model, x_client, y_client, protected, privileged) :
    if protected == 'Female' or privileged == 'Female' :
        protected_x = x_client[x_client['sex']==1.0] #correspond a female
        protected_y = y_client[x_client['sex'] == 1.0]

        privileged_x = x_client[x_client['sex']==0.0]
        privileged_y = y_client[x_client['sex'] == 0.0]

    else :
        protected_x = x_client[x_client[protected]==1.0]
        protected_y = y_client[x_client[protected] == 1.0]

        privileged_x = x_client[x_client[privileged]==1.0]
        privileged_y = y_client[x_client[privileged] == 1.0]

    tpr_protected = compute_TPR(protected_x, protected_y, model)
    tpr_privilieged = compute_TPR(privileged_x, privileged_y, model)

    # P(\hat{Y} = 1 | Y = 1, S=s_1) - P(\hat{Y} = 1 | Y = 1, S=s_2)
    return tpr_privilieged - tpr_protected

def SPD(model, x_client, y_client, protected, privileged) :

    if protected == 'Female' or privileged == 'Female' :
        protected_x = x_client[x_client['sex']==1.0] #correspond a female
        protected_y = y_client[x_client['sex'] == 1.0]

        privileged_x = x_client[x_client['sex']==0.0]
        privileged_y = y_client[x_client['sex'] == 0.0]
    else :
        protected_x = x_client[x_client[protected]==1.0]
        protected_y = y_client[x_client[protected]==1.0]

        privileged_x = x_client[x_client[privileged]==1.0]
        privileged_y = y_client[x_client[privileged]==1.0]

    pp_protected = compute_PP(protected_x, protected_y, model)
    pp_privileged = compute_PP(privileged_x, privileged_y, model)
    return pp_privileged - pp_protected



def plot_Fairness_Values(model, x_client, y_client, sensitive_attr, model_id, iteration) :

    eod_values = []
    spd_values = []
    labels = []

    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(sensitive_attr)) :
        for j in range(i, len(sensitive_attr)) :
            if sensitive_attr[i] != sensitive_attr[j] :

                labels.append(sensitive_attr[i]+'/'+sensitive_attr[j])
                eod_values.append(EOD(model, x_client, y_client, sensitive_attr[i], sensitive_attr[j]))
                spd_values.append(SPD(model, x_client, y_client, sensitive_attr[i], sensitive_attr[j]))
            #    offset = width * multiplier
    x_axis = np.arange(len(labels))
    rects = ax.bar(x_axis - 0.1, eod_values, 0.10, label='EOD')
    ax.bar_label(rects, padding=3)
    rects = ax.bar(x_axis + 0.1, spd_values, 0.10, label='SPD')
    ax.bar_label(rects, padding=3)
    ax.axhline(y=0.0, color='r', linestyle='-')
    multiplier += 1

    # plots
    x_locations = np.arange(len(eod_values))  # the label locations
    ax.set_ylabel('Valeurs')
    ax.set_title('Equal opportunity / Statistical Parity sur les differents groupes [Model : '+model_id+', iteration : '+iteration+']')
    ax.set_xticks(x_locations + width, labels, rotation=45)
    ax.legend(loc='upper left', )
    ax.set_ylim(-1, 1)
    return fig


def Eval_group_fairness(model, x, y, sensitive_attr, model_id, iteration) :
    groups_x = []
    groups_y = []
    if sensitive_attr[0] != 'Male' and sensitive_attr[1] != 'Male':
        #partitionnement
        for i in range (len(sensitive_attr)) :
            print("evaluating for group : ", sensitive_attr[i])
            groups_x.append(x[x[sensitive_attr[i]] == 1.0])
            groups_y.append(y[x[sensitive_attr[i]] == 1.0])
    #Male/Female est un attribut binaire et est traité différemment dans le preprocess
    # -> une seule colomne est gardée (Female) pour des valeurs 0/1
    else:
        groups_x.append(x[x['sex'] == 1.0]) #correspond a female
        groups_x.append(x[x['sex'] == 0.0])

        groups_y.append(y[x['sex'] == 1.0])
        groups_y.append(y[x['sex'] == 0.0])

    group_evals = []
    metrics = ("Loss", "Accuracy", "Precision", "Recall/TPR", "PositiveProportion")
    for i in range (len(sensitive_attr)) :
        print ("Performance du modele sur le groupe ("+sensitive_attr[i]+" = 1)")
        group_evals.append(model.evaluate(groups_x[i], groups_y[i], verbose=2))
        group_evals[i].append(compute_PP(groups_x[i], groups_y[i], model)) #Ajouter la métrique faite en locale


    x = np.arange(len(metrics))  # the label locations
    width = 0.10  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for i in range (len (sensitive_attr)) :
        offset = width * multiplier
        rects = ax.bar(x + offset, group_evals[i], width, label=sensitive_attr[i])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # plots
    ax.set_ylabel('Valeurs')
    ax.set_title('Performance du modele sur les differents groupes [Model : '+model_id+', iteration : '+iteration+']')
    ax.set_xticks(x + width, metrics)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.5)
    return fig

"""
Data-set unfairness : 1) Disparate impact : measure correlations between unprotected and the protected attr
                      2) Disparate Treatmennt : measure correlation between label and protected attr

"""


def disparate_treatment(x, y, protected, privileged, label) :
    if protected == 'Female' or privileged == 'Female' :
        #protected_x = x_client[x_client['sex']==1.0] #correspond a female
        protected_y = y[x['sex'] == 1.0]

        #privileged_x = x_client[x_client['sex']==0.0]
        privileged_y = y[x['sex'] == 0.0]
    else :
        #protected_x = x_client[x_client[protected]==1.0]
        protected_y = y[x[protected]==1.0]

        #privileged_x = x_client[x_client[privileged]==1.0]
        privileged_y = y[x[privileged]==1.0]

    #protected_y = y[x[protected_group] == 1.0]
    #privileged_y = y[x[privileged_group] == 1.0]
    positive_pred_prop_protected = protected_y.mean()
    print(protected, 'positive pred : ', positive_pred_prop_protected)
    positive_pred_prop_privileged = privileged_y.mean()
    print(privileged, 'positive pred : ', positive_pred_prop_privileged)
    return positive_pred_prop_privileged/positive_pred_prop_protected


def disparate_impact(data, sensitive_attr) :
    #The goal is to train a discriminator that predicts the sens_attr from non-sensitive ones
    #and measure its balanced error rate BER.

    y = data[sensitive_attr]
    x = data.drop(sensitive_attr, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    print('training an adverserial classifier to measure disparate impact ...')
    discriminator = Adult_NN((None ,x.shape[1]))
    discriminator.fit(x, y, epochs=50, verbose=0)
    eval = discriminator.evaluate(x_test, y_test)
    fpr = 1 - eval[1] # 1 - precision
    fnr = 1 - eval[2] # 1 - recall
    ber = (fpr + fnr)/2
    print('BER of discriminator is : ', ber)
    return ber
