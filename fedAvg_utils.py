import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import sys
import tensorflow as tf
import asyncio
import subsetsum as ss
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import sklearn
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
#from scikeras.wrappers import KerasClassifier
#from art.estimators.classification import KerasClassifier
import json
from adult_utils import *
from fairness_utils import *
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


sys.path.insert(0, os.path.abspath('..'))
#tf.compat.v1.disable_eager_execution()

#tf.random.set_seed(1234)
random.seed(datetime.now().timestamp())
SEED=random.randint(0, 123494321)

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



set_global_determinism()

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum([tf.convert_to_tensor(grad_list_tuple[i]) for i in range(len(scaled_weight_list))] , axis=0)
        avg_grad.append(layer_mean)

    return avg_grad

#Two versions :
#.numpy() to convert a tensor to a numpy array doesnt work when eager_exection is disabled
#repace with .eval() after creating a tensorflow Session()

def FedAvg_eager_exec_disabled(models, n, clients_weights, input_shape) :
    scaled_weights = []

    global_model = Adult_NN(input_shape, 'zeros')
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))
    avg_weights = sum_scaled_weights(scaled_weights)
    with tf.compat.v1.Session() as sess :
        raw_weights = [avg_weight_layer.eval(session=sess) for avg_weight_layer in avg_weights]
    global_model.set_weights(raw_weights)
    #global_model.set_weights(avg_weights)
    return global_model


def FedAvg(models, n, clients_weights, input_shape) :
    scaled_weights = []

    global_model = Adult_NN(input_shape, 'zeros')
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))
    avg_weights = sum_scaled_weights(scaled_weights)
    global_model.set_weights([avg_weight_layer.numpy() for avg_weight_layer in avg_weights])
    #global_model.set_weights(avg_weights)
    return global_model



#Initialization rule :
# if activation function is sigmoid or Tanh ==> use Glorot (Xavier or normalized Xavier)
# if activation is Relu ==> use He initialization from He et Al 2015

def Adult_NN(input_shape, init_distrib) :
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer = init_distrib, kernel_initializer= init_distrib)(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Precision(name='Precision')

    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
        #run_eagerly=True
    )
    return model


def train_from_model(model, x, y, epch, client_id) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    #10-Fold cross validation
    #kf = KFold(n_plits=10)
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epch, verbose=1)
    plot_learningCurve(history, epch, client_id)
    model.evaluate(x_test, y_test)
    return model



def update_local_model(agg_model, input_shape) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(agg_model)
    local_model.build(input_shape)
    local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
                ]
            )

    local_model.set_weights(agg_model.get_weights())
    return local_model
