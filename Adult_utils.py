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
import math
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

SEED=random.randint(0, 123494321)



"""
A model architecture that achives good performance on Adult

"""

def Adult_NN(input_shape) :
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED))(inputs)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='recall')

    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model


def train_from_model(model, x, y, epch, client_id) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    #10-Fold cross validation
    #kf = KFold(n_plits=10)
    print(x_train.shape)
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epch, verbose=1)
    plot_learningCurve(history, epch, client_id)
    model.evaluate(x_test, y_test)
    return model

def plot_learningCurve(history, epoch, client_id):
    # Plot training & validation accuracy values
    epoch_range = range(1, epoch+1)

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(epoch_range, history.history['accuracy'])
    axis[0].plot(epoch_range, history.history['val_accuracy'])
    axis[0].set_title("Client "+ str(client_id)+ ": Model accuracy")
    axis[0].set_ylabel('Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].legend(['Train', 'Val'], loc='upper left')
    #plt.show()

    # Plot training & validation loss
    axis[1].plot(epoch_range, history.history['loss'])
    axis[1].plot(epoch_range, history.history['val_loss'])
    axis[1].set_title("Client "+ str(client_id)+ ": Model loss")
    axis[1].set_ylabel('Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def dataSet_summary(data) :
    print(data.size, "entry in this Dataset")
    print(data.head())
    print(data.shape)
    print ("Null values check : ")
    print ("\n", data.isnull().sum())
    print ("\n \nBalance check : ")

    class_1 = data[data['income'] == '>50K']
    class_0 = data[data['income'] == '<=50K']
    ratio = min(class_0.size / class_1.size, class_1.size / class_0.size)
    print("balance level : ", round(ratio, 5) * 100, "%")

""""
attributes est un dictionnaire dont les clés sont les noms des colonnes sont les valeurs possibles
"""

def check_balance(data, attributes) :

    count=0
    plt.rcdefaults()
    fig, ax = plt.subplots(len(attributes), 1)
    for column in attributes.keys() :
        values = attributes[column]
        rates = []
        #l'attribit sex est traité différemment
        if column == 'sex' :
            rate = 100 * round(data[data['sex'] == 1.0].size / data.size, 2)
            rates.append(rate)
            print("Male rate is : "+ str(rate))
            rate = 100 * round(data[data['sex'] == 0.0].size / data.size, 2)
            rates.append(rate)
            print("Female rate is : "+ str(rate))
            values = ['Male', 'Female']
        if column == 'income' :
            rate = 100 * round(data[data['income'] == 1.0].size / data.size, 2)
            rates.append(rate)
            print(">=50K rate is : "+ str(rate))
            rate = 100 * round(data[data['income'] == 0.0].size / data.size, 2)
            rates.append(rate)
            print("<50K rate is : "+ str(rate))
            values = ['>=50K', '<50K']

        if column != 'sex' and column != 'income' :
            for j in range (len(values)) :
                if values[j] != '?' :
                    rate = 100 * round(data[data[values[j]] == 1.0].size / data.size , 2)
                    rates.append(rate)
                    print (str(values[j]) + " rate is : ", str(rate))
                else :
                    rates.append(0.0)


        y_pos = np.arange(len(values))
        error = np.random.rand(len(values))
        ax[count].barh(y_pos, rates, xerr=error, align='center')
        ax[count].set_yticks(y_pos, labels=values)
        ax[count].invert_yaxis()  # labels read top-to-bottom
        if count==(len(attributes.keys())-1) :
            ax[count].set_xlabel('Taux')
        ax[count].set_title('La distribution des valeurs pour "' +column+'"')
        count += 1
    return fig


def binary_encode(data, columns) :

    label_encoder = LabelEncoder()
    for column in columns :
        data[column] = label_encoder.fit_transform(data[column])
    return data

def onehot_encode(data, columns) :

    for column in columns :
        dummies = pd.get_dummies(data[column])
        data = pd.concat([data, dummies], axis=1)
        data.drop(column, axis=1, inplace=True)

    return data
"""
    Remplacer les '?' par np.NaN et Encoder les attributs de catégorie en attributs numériques

"""
def data_PreProcess(data) :
    data = data.replace('?', np.NaN)
    data.drop('education', axis=1, inplace=True)
    nominal_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
    binary_features = ['sex']

    data = onehot_encode(data, nominal_features)
    data = binary_encode(data, binary_features)
    #y = data['income']
    #x = data.drop('income', axis=1)
    #<= 50K ---> 0 et > 50K ---> 1
    label_encoder = LabelEncoder()
    #y = label_encoder.fit_transform(y)
    data['income'] = label_encoder.fit_transform(data['income'])
    #mapping de toutes les valeurs numériques vers l'intervalle [0, 1]
    #normalized_x = (x - x.min()) / (x.max() - x.min())
    normalized_data = (data - data.min()) / (data.max() - data.min())
    #return pd.concat([normalized_x, y], axis=1)
    return normalized_data

def create_alphas(dim) :
    alphas = {
        'extremely homogeneous'   : [100000     for i in range(dim)],
        'very homogeneous'        : [1000       for i in range(dim)],
        'homogeneous'             : [10         for i in range(dim)],
        'uniform'                 : [1          for i in range(dim)], #Eqivaut a un echantillonage uniforme sample(random_state = 0)
        'heterogeneous_2'         : [1/2        for i in range(dim)],
        'heterogeneous_5'         : [1/5        for i in range(dim)],
        'heterogeneous_10'        : [1/10       for i in range(dim)],
        'very heterogeneous'      : [1/100      for i in range(dim)],
        'extremely heterogeneous' : [1/1000     for i in range(dim)] #inutilisable -> Des valeurs trop petite (2^-128=> np.NaN)
    }
    return alphas

def Dirichelet_sampling(data, alphas, values,  n_lines) :
    #values = data[col].unique()
    assert (n_lines <= len(data.axes[0]))
    assert (len(values) == len(alphas))
    print (len(data.axes[0]))
    #sample a distribution from dir(alphas)
    s = np.random.dirichlet(tuple(alphas), 1).tolist()[0]
    print (s)
    print (alphas)
    groups = []
    for i in range (len(values)) :
        if values[0] == 'Male' or values[0] == 'Female' :
            if math.isnan(s[0]) :
                s[0] = 1/n_lines
            if round(n_lines * s[0]) > 0:
                groups.append(data[data['sex'] == 1.0].sample(n=round(n_lines * s[0]), replace=True))
            else:
                groups.append(data[data['sex'] == 1.0].sample(n=1))
            if math.isnan(s[1]) :
                s[1] = 1/n_lines
            if round(n_lines * s[1]) > 0:
                groups.append(data[data['sex'] == 0.0].sample(n=round(n_lines * s[0]), replace=True))
            else:
                groups.append(data[data['sex'] == 1.0].sample(n=1))

        else :
            if round(n_lines * s[i]) > 0 :
                groups.append(data[data[values[i]]==1.0].sample(n=round(n_lines * s[i]), replace=True))
            else :
                groups.append(data[data[values[i]] == 1.0].sample(n=1))

    return pd.concat(groups)

# splitting x into n_client pieces of size client_dataset_size rows
def dataset_slice(data, client_dataset_size, token) :
    dataset_cols = len(data.axes[1])
    d_client = data.iloc[token * client_dataset_size : (token+1) * client_dataset_size, 0 : dataset_cols]
    return d_client



def save_model(model, name, iteration) :
    loc = '../SharedFiles/Client_Models/Client'+str(name)+'/iteration='+str(iteration) # save location
    print('saving model at location : ', loc)
    try :
        os.mkdir(loc)
    except OSError:
        print ("[save_model-error] Erreur lors de l'ouverture du fichier")
        sys.exit()

    count = 0
    for layer in model.layers:
        if layer.get_weights() != []:
            np.savetxt(loc + '/'+"dense_"+str(count)+"_kernel.csv", layer.get_weights()[0].flatten(), delimiter=",")
            np.savetxt(loc + '/'+"dense_"+str(count)+"_bias.csv", layer.get_weights()[1].flatten(), delimiter=",")
        count+=1


def save_metric(metric_value, name, iteration) :
    try :
        f = open('../SharedFiles/Fairness_Metrics/Client'+str(name)+'/iteration='+str(iteration)+'.txt', "w+")
    except OSError:
        print ("[save_metric-error] Erreur lors de l'ouverture du fichier")
        sys.exit()
    f.write(str(metric_value))
    f.close()


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
