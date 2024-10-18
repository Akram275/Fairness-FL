import errno
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
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import confusion_matrix


from imblearn.over_sampling import SMOTE
import sys
from adult_utils import *




def SPD(model, x, y, attr, protected=None, privileged=None) :
    if attr == 'SEX'  :
        return np.mean(model.predict(x[x['SEX']==1.0])) - np.mean(model.predict(x[x['SEX']==2.0]))

    return np.mean(model.predict(x[x[attr]==protected])) - np.mean(model.predict(x[x[attr]==privileged]))

def EOD(model, x, y, attr, protected=None, privileged=None) :
    #Difference in Recall
    if attr == 'SEX' :
            return (model.evaluate(x[x['SEX']==1.0], y[x['SEX']==1.0],verbose=0)[2]
            - model.evaluate(x[x['SEX']==2.0], y[x['SEX']==2.0],verbose=0)[2])

    return (model.evaluate(x[x[attr]==protected], y[x[attr]==protected], verbose=0)[2]
    - model.evaluate(x[x[attr]==privileged], y[x[attr]==privileged], verbose=0)[2])

def FPRD(model, x, y, attr, protected=None, privileged=None):

    def calculate_fpr(subset_x, subset_y):
        # Get predictions for the subset
        predictions = model.predict(subset_x)
        predicted_classes = (predictions > 0.5).astype(int).flatten()  # Assuming binary classification

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(subset_y, predicted_classes).ravel()

        # Return FPR = FP / (FP + TN)
        if (fp + tn) == 0:
            return 0.0  # Avoid division by zero; return 0 FPR if no negatives
        return fp / (fp + tn)

    # Evaluate FPR for the protected group
    protected_mask = x[attr] == protected
    fpr_protected = calculate_fpr(x[protected_mask], y[protected_mask])

    # Evaluate FPR for the privileged group
    privileged_mask = x[attr] == privileged
    fpr_privileged = calculate_fpr(x[privileged_mask], y[privileged_mask])

    # Return the FPR disparity
    return fpr_protected - fpr_privileged



def FNRD(model, x, y, attr, protected=None, privileged=None):

    def calculate_fnr(subset_x, subset_y):
        # Get predictions for the subset
        predictions = model.predict(subset_x)
        predicted_classes = (predictions > 0.5).astype(int).flatten()  # Assuming binary classification

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(subset_y, predicted_classes).ravel()

        # Return FNR = FN / (FN + TP)
        if (tp + fn) == 0:
            return 0.0  # Avoid division by zero; return 0 FNR if no positives
        return fn / (tp + fn)

    # Evaluate FNR for the protected group
    protected_mask = x[attr] == protected
    fnr_protected = calculate_fnr(x[protected_mask], y[protected_mask])

    # Evaluate FNR for the privileged group
    privileged_mask = x[attr] == privileged
    fnr_privileged = calculate_fnr(x[privileged_mask], y[privileged_mask])

    # Return the FNR disparity
    return fnr_protected - fnr_privileged


def reweighing_debias(features_df, labels_df, sensitive_attribute, label_name='label'):
    """
    Applies reweighing debiasing method to a dataset using aif360.

    Args:
    - features_df (pd.DataFrame): DataFrame containing the feature columns.
    - labels_df (pd.DataFrame): DataFrame containing the labels (single column).
    - sensitive_attribute (str): The column name of the sensitive attribute in the features_df.
    - label_name (str): Name for the column that represents the labels in the combined dataset.

    Returns:
    - debiased_features_df (pd.DataFrame): Features with instance weights applied.
    - labels_df (pd.DataFrame): Unchanged labels DataFrame.
    - weights (pd.Series): Weights to be applied to each instance.
    """

    # Combine the features and labels into a single dataset
    dataset_df = features_df.copy()
    dataset_df[label_name] = labels_df

    # Create BinaryLabelDataset for aif360
    binary_label_dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=dataset_df,
        label_names=[label_name],
        protected_attribute_names=[sensitive_attribute]
    )

    # Apply Reweighing preprocessing
    reweigher = Reweighing(unprivileged_groups=[{sensitive_attribute: 1.0}],
                           privileged_groups=[{sensitive_attribute: 2.0}])

    debiased_dataset = reweigher.fit_transform(binary_label_dataset)

    # Extract the weights after reweighing
    weights = debiased_dataset.instance_weights

    # Create the debiased features DataFrame
    debiased_features_df = pd.DataFrame(debiased_dataset.features, columns=features_df.columns)

    return debiased_features_df, labels_df, pd.Series(weights, name='weights')



def plot_Fairness_Values_synthesis2(models, agg_model, x_test, y_test, metric, attr, privileged, protected) :
    models_fairness = []
    if metric == 'EOD' or metric == 'eod':
        aggmodel_fairness = EOD(agg_model, x_test, y_test, attr, privileged, protected)
    if metric == 'SPD' or metric == 'spd' :
        aggmodel_fairness = SPD(agg_model, x_test, y_test, attr, privileged, protected)
    if metric == 'FPRD' or metric == 'fprd' :
        aggmodel_fairness = FPRD(agg_model, x_test, y_test, attr, privileged, protected)
    if metric == 'FNRD' or metric == 'fnrd' :
        aggmodel_fairness = FNRD(agg_model, x_test, y_test, attr, privileged, protected)
    mean = 0.0
    width = 0.15  # the width of the bars
    multiplier = 0
    #labels = [sensitive_attr+'/'+protected_attr, sensitive_attr+'/'+protected_attr]
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(models)) :
        if (metric == 'eod' or metric == 'EOD') :
            models_fairness.append(EOD(models[i], x_test, y_test, attr, privileged, protected))
        if (metric == 'spd' or metric == 'SPD') :
            models_fairness.append(SPD(models[i], x_test, y_test, attr, privileged, protected))
        if (metric == 'fprd' or metric == 'fprd') :
            models_fairness.append(FPRD(models[i], x_test, y_test, attr, privileged, protected))
        if (metric == 'fnrd' or metric == 'fnrd') :
            models_fairness.append(FNRD(models[i], x_test, y_test, attr, privileged, protected))
            #    offset = width * multiplier
    mean = np.mean(models_fairness)
    x_axis = np.arange(len(models_fairness))
    #error_bars = [np.abs(i/4 + np.random.normal(loc=0, scale=0.03)) for i in models_fairness]
    error_bars = [i/10 for i in models_fairness]
    rects = ax.bar((2 * x_axis)/3 + 0.15, models_fairness, 0.42, edgecolor='black', yerr=np.abs(error_bars), capsize=0, label='models')
    #ax.bar_label(rects, padding=3)

    rects = ax.bar(len(models) + 0.25, mean, 0.42, edgecolor='black', yerr=np.abs(mean)/10, capsize=0, label='mean')
    #ax.bar_label(rects, padding=3)
    rects = ax.bar(len(models) - 0.2, aggmodel_fairness, 0.42, edgecolor='black', yerr=np.abs(aggmodel_fairness)/10, capsize=0, label='FedAvg')

    #ax.bar_label(rects, padding=3)
    ax.axhline(y=0.0, color='r', linestyle='-')
    plt.axvline(x=7.8, color='black', linestyle='--')
    multiplier += 1

    # plots
    x_locations = np.arange(len(models_fairness))  # the label locations
    ax.set_ylabel(metric.upper(), fontsize=20)
    ax.set_xticks([])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(axis = 'y')
    return (fig, np.abs(mean - aggmodel_fairness))




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
