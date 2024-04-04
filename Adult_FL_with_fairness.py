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
from art.estimators.classification import KerasClassifier

from Adult_utils import *
from fairness_utils import *
from sklearn.linear_model import LinearRegression
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.estimators.regression.keras import KerasRegressor
from art.utils import load_diabetes
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference import ShadowModels
from art.utils import to_categorical
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased


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

#Subset sum problem with a parameter epsilon for approximations
def optimal_subset(fairness_values, max_attempts) :
    epsilone = 5
    n_attempts = 0
    while(not ss.has_solution(fairness_values, epsilone)) :
        epsilone+=5
        n_attempts+=1
        #max attempts reached : return false and all indices
        if n_attempts == max_attempts :
            print('[optimal_Subset] Final attempt failure ')
            return (False, [i for i in range(len(fairness_values))])

    print(f'[optimal_Subset] Found subset for epsilone = {round(epsilone/1000, 3)} at attempt n° {n_attempts}')
    solutions = []
    solutions_size = []
    for s in ss.solutions(fairness_values, epsilone) :
        solutions.append(s)
        solutions_size.append(len(s))

    #return the biggest one
    subset = solutions[np.argmax(solutions_size)]

    return (True, subset)

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


#A bit useless
def df_to_series(dataframe) :
    series = []
    for i in x_test.columns :
        series.append(x_train[i].squeeze())
    return pd.concat(series, axis=1)


def update_local_model(agg_model, input_shape) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(Agg_model)
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


def plot_Fairness_Values_synthesis(model1, model2, agg_model, x_test, y_test, sensitive_attr, metric) :
    model1_fairness = []
    model2_fairness = []
    aggmodel_fairness = []
    labels = []

    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(sensitive_attr)) :
        for j in range(i, len(sensitive_attr)) :
            if sensitive_attr[i] != sensitive_attr[j] :

                labels.append(sensitive_attr[i]+'/'+sensitive_attr[j])

                if (metric == 'eod' or metric == 'EOD') :
                    model1_fairness.append(EOD(model1, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    model2_fairness.append(EOD(model2, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    aggmodel_fairness.append(EOD(agg_model, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                if (metric == 'spd' or metric == 'SPD') :
                    model1_fairness.append(SPD(model1, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    model2_fairness.append(SPD(model2, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    aggmodel_fairness.append(SPD(agg_model, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
            #    offset = width * multiplier
    x_axis = np.arange(len(labels))
    rects = ax.bar(x_axis - 0.1, model1_fairness, 0.10, label='model1')
    ax.bar_label(rects, padding=3)
    rects = ax.bar(x_axis + 0.1, model2_fairness, 0.10, label='model2')
    ax.bar_label(rects, padding=3)
    rects = ax.bar(x_axis + 0.0, aggmodel_fairness, 0.10, label='agg_model')
    ax.bar_label(rects, padding=3)

    ax.axhline(y=0.0, color='r', linestyle='-')
    multiplier += 1

    # plots
    x_locations = np.arange(len(model1_fairness))  # the label locations
    ax.set_ylabel(metric)
    #ax.set_title('models fairness evaluations')
    ax.set_xticks(x_locations + width, labels, rotation=45)
    ax.legend(loc='upper left', )
    ax.set_ylim(-1, 1)
    return fig




def plot_Fairness_Values_synthesis2(models, agg_model, x_test, y_test, sensitive_attr, protected_attr, metric, opt=False, indices=None) :
    models_fairness = []
    if metric == 'EOD' or metric == 'eod':
        aggmodel_fairness = EOD(agg_model, x_test, y_test, sensitive_attr, protected_attr)
    if metric == 'SPD' or metric == 'spd' :
        aggmodel_fairness = SPD(agg_model, x_test, y_test, sensitive_attr, protected_attr)
    mean = 0.0
    width = 0.15  # the width of the bars
    multiplier = 0
    labels = [sensitive_attr+'/'+protected_attr, sensitive_attr+'/'+protected_attr]
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(models)) :
        if (metric == 'eod' or metric == 'EOD') :
            models_fairness.append(EOD(models[i], x_test, y_test, sensitive_attr, protected_attr))
        if (metric == 'spd' or metric == 'SPD') :
            models_fairness.append(SPD(models[i], x_test, y_test, sensitive_attr, protected_attr))
            #    offset = width * multiplier
    mean = np.mean(models_fairness)
    x_axis = np.arange(len(models_fairness))
    rects = ax.bar((2 * x_axis)/3 + 0.2, models_fairness, 0.20, label='models')
    ax.bar_label(rects, padding=3)

    rects = ax.bar(len(models) + 0.20, mean, 0.20, label='mean')
    ax.bar_label(rects, padding=3)

    rects = ax.bar(len(models) + 0.0, aggmodel_fairness, 0.20, label='aggregated model (FedAvg)')
    ax.bar_label(rects, padding=3)

    ax.axhline(y=0.0, color='r', linestyle='-')
    multiplier += 1

    # plots
    x_locations = np.arange(len(models_fairness))  # the label locations
    ax.set_ylabel(metric)
    ax.set_xticks([])
    #ax.set_title('models fairness evaluations')
    #ax.set_xticks([0, 6], labels, rotation=45)
    ax.legend(loc='upper left', )
    ax.set_ylim(-0.75, 0.75)
    return fig


if __name__ == '__main__':


    data = pd.read_csv('datasets/adult.csv')

    attributes = {
            'race'      : data['race'].unique(),
            'sex'       : data['sex'].unique(),
         #   'relationship' : data['relationship'].unique(),
         #   'occupation' : data['occupation'].unique(),
            'marital.status' : data['marital.status'].unique(),
            'income' : data['income'].unique()
    }

    data.replace('?', np.NaN)

    print("Data preprocessing ...")
    pre_processed_data = data_PreProcess(data)

    dataset_cols = len(pre_processed_data.axes[1])
    dataset_rows = len(data.axes[0])

    #training_data = pre_processed_data.iloc[0 : round(0.7 * dataset_rows),                0 : dataset_cols]
    #shadow_attack_data   = pre_processed_data.iloc[round(0.3 * dataset_rows) + 1 : dataset_rows, 0 : dataset_cols]

    training_data = pre_processed_data.sample(frac=0.7)
    shadow_attack_data = pre_processed_data.drop(training_data.index)

    y_train = training_data['income']
    x_train = training_data.drop('income', axis=1)
    #splitting training_dataset in half for the two clients
    alphas = create_alphas(len(data['race'].unique()))
    y_shadow = shadow_attack_data['income']
    x_shadow = shadow_attack_data.drop('income', axis=1)


    input_shape = (x_shadow.shape[1],)

    n_clients = 5
    target_loss = 3.0
    target_acc = 0.83
    target_recall = 0.6
    target_precision = 0.6
    epochs = 80
    n_iterations = 0
    max_iterations = 10
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores = []
    Agg_model = Adult_NN(input_shape, init_distrib)
    d_clients = []
    for i in range(n_clients) :
        d_clients.append(Dirichelet_sampling(training_data, alphas['heterogeneous_2'], data['race'].unique(), 20000))

    while (True) :
        models = []
        n_iterations+=1
        fairness_values = []
        for i in range(n_clients) :
            y_client = d_clients[i]['income']
            x_client = d_clients[i].drop('income', axis=1)
            model = update_local_model(Agg_model, input_shape)
            print('training model n°', i+1)
            history = model.fit(x_client, y_client, validation_split=0.2, batch_size=32, epochs=epochs, verbose=0)
            models.append(model)
            #plot_learningCurve(history, epochs, n_iterations)
            fairness_values.append(round(SPD(models[i], x_client, y_client, 'Black', 'Other') * 1000))

        opt, indices = optimal_subset(fairness_values, 10)
        if opt :
            print('selected clients ', indices)
        else :
            print('optimal subset selection failed after max_attempts tries')
        Agg_model = FedAvg(models, n_clients, [round(1/n_clients, 2) for i in range(n_clients)], input_shape)
        synthesis2 = plot_Fairness_Values_synthesis2(models, Agg_model, x_train, y_train, 'Black', 'Other', 'SPD')
        #Replace 'Some_file' by your file
        synthesis2.savefig('Some_file/iteration_'+str(n_iterations))
        #plt.show(block=False)
        #Two options : evaluate agg_model over the whole dataset or only over the shadow data
        scores.append(Agg_model.evaluate(x_train, y_train, verbose=0))
        print('Agg model progress so far [Loss, Acc, Recall, Precision] (Over union dataset) : \n')

        for i in range(len(scores)) :
            print(f"iteration {i} : {[round(s, 5) for s in scores[i]]}")

        if ( n_iterations >= max_iterations                   or
            ( scores[n_iterations-1][0] <= target_loss        and
            scores[n_iterations-1][1] >= target_acc           and
            scores[n_iterations-1][2] >= target_recall        and
            scores[n_iterations-1][3] >= target_precision)) :
            break


