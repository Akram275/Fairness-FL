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
from Adult_utils import *
from fairness_utils import *
from sklearn.linear_model import LinearRegression
#from art.estimators.regression.scikitlearn import ScikitlearnRegressor
#from art.estimators.regression.keras import KerasRegressor
#from art.utils import load_diabetes
#from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
#from art.attacks.inference.membership_inference import ShadowModels
#from art.utils import to_categorical
#from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

#from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased


from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.optimize import linprog

#from aif360.datasets import BinaryLabelDataset
#from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
#from aif360.metrics import BinaryLabelDatasetMetric
#from aif360.metrics import ClassificationMetric
#from aif360.algorithms.preprocessing.reweighing import Reweighing


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



def Optimize_weights(fairness) :
    # Assuming 'fairness' is the set of  fairness eval (EOD or SPD or others )
    fairness_plus = np.maximum(0, fairness)  # Positive subset
    fairness_minus = -np.minimum(0, fairness)  # Negative subset


    def objective_function(weights, n):
        return np.sum((weights - 1/n)**2)

    #sum weights = 1
    def constraint_function(weights):
        return np.sum(weights) - 1

    # Custom equality function: weighted sum of positive numbers equals the absolute value of the weighted sum of negative numbers
    def equality_function(weights):
        return np.sum(weights * fairness_plus) - np.sum(np.abs(weights * fairness_minus))

    # Constraint: weights are non-negative
    bounds = [(0, 1) for _ in range(len(fairness))]

    # Initial guess for weights
    initial_weights = np.ones(len(fairness)) / len(fairness)

    # Solve the optimization problem
    result = minimize(objective_function, initial_weights, args=(len(fairness),), method='SLSQP', constraints=[
        {'type': 'eq', 'fun': equality_function},
        {'type': 'eq', 'fun': constraint_function},
    ], bounds=bounds)


    # Extract the optimal weights
    optimal_weights = result.x
    return optimal_weights




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



def df_to_series(dataframe) :
    series = []
    for i in x_test.columns :
        series.append(x_train[i].squeeze())
    return pd.concat(series, axis=1)

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall



#model is a regressor ==> use 'loss' as input_types
#model is a Classifier ==> use 'prediction' as input_types
# works with attack_model_type = 'gb' (Gradient boosting) and attack_model_type = 'rf'  (random forest)
# error when attack_model_type ='nn' (Neural network) is used

def black_box_MIA(model, x_train, y_train, x_test, y_test, attack_model_type='rf', input_type='prediction') :
    art_classifier = KerasClassifier(model=model)
    #art_classifier.fit(x_train, y_train, nb_epochs=epochs, batch_size=32, verbose=1)
    print('Base model score: ', model.evaluate(x_test, y_test))
    #model is a regressor ==> use 'loss' as input_types
    #model is a Classifier ==> use 'prediction' as input_types
    # works with attack_model_type = 'gb' (Gradient boostring) and attack_model_type = 'rf'  (random forest)
    # error when attack_model_type ='nn' (Neural network) is used
    bb_attack = MembershipInferenceBlackBox(art_classifier, attack_model_type=attack_model_type, input_type=input_type)
    bb_attack.fit(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy(), verbose=1)
    # infer
    inferred_train_bb = bb_attack.infer(x_train.astype(np.float32), y_train)
    inferred_test_bb = bb_attack.infer(x_test.astype(np.float32), y_test)

    # check accuracy
    train_acc_bb = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc_bb = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc_bb = (train_acc_bb * len(inferred_train_bb) + test_acc_bb * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print('Member accuracy:', train_acc_bb)
    print('Non-Member accuracy:', test_acc_bb)
    print('Accuracy:', acc_bb)
    # rule-based
    print(calc_precision_recall(np.concatenate((inferred_train_bb, inferred_test_bb)),
                                np.concatenate((np.ones(len(inferred_train_bb)), np.zeros(len(inferred_test_bb))))))
    return [train_acc_bb, test_acc_bb, acc_bb]

def Shadow_models_MIA(model, x_train, y_train, x_test, y_test, x_shadow, y_shadow) :
    print('art classifier creation')
    art_classifier = KerasClassifier(model, use_logits=False)
    shadow_models = ShadowModels(art_classifier, num_shadow_models=3, disjoint_datasets=True, random_state=1)
    shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, y_shadow)
    (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

    # Shadow models' accuracy
    print([sm.model.evaluate(x_test, y_test) for sm in shadow_models.get_shadow_models()])
    # rf (random forest) or gb (gradient boosting)
    attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="gb")
    attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions, epochs=100)

    member_infer = attack.infer(x_train, y_train)
    nonmember_infer = attack.infer(x_test, y_test)
    member_acc = np.sum(member_infer) / len(x_train)
    nonmember_acc = 1 - np.sum(nonmember_infer) / len(x_test)
    acc = (member_acc * len(x_train) + nonmember_acc * len(x_test)) / (len(x_train) + len(x_test))
    print('Attack Member Acc:', member_acc)
    print('Attack Non-Member Acc:', nonmember_acc)
    print('Attack Accuracy:', acc)
    return [member_acc, nonmember_acc, acc]

def RuleBase_MIA(model, x_train, y_train, x_test, y_test) :
    art_classifier = KerasClassifier(model)
    rulebase_attack = MembershipInferenceBlackBoxRuleBased(art_classifier)
    mlp_inferred_train = rulebase_attack.infer(x_train.astype(np.float32), y_train)
    mlp_inferred_test = rulebase_attack.infer(x_test.astype(np.float32), y_test)

    mlp_train_acc = np.sum(mlp_inferred_train) / len(mlp_inferred_train)
    mlp_test_acc = 1 - (np.sum(mlp_inferred_test) / len(mlp_inferred_test))
    mlp_acc = (mlp_train_acc * len(mlp_inferred_train) + mlp_test_acc * len(mlp_inferred_test)) / (len(mlp_inferred_train) + len(mlp_inferred_test))
    print('member acc : ', mlp_train_acc)
    print('non member acc : ', mlp_test_acc)
    print('overall acc : ', mlp_acc)

    print(calc_precision_recall(np.concatenate((mlp_inferred_train, mlp_inferred_test)),
                            np.concatenate((np.ones(len(mlp_inferred_train)), np.zeros(len(mlp_inferred_test))))))

    return [mlp_train_acc, mlp_test_acc, mlp_acc]

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
                lab1 = sensitive_attr[i]
                lab2 = sensitive_attr[j]
                #A more compact labelization for a better plot
                if sensitive_attr[i] == 'Asian-Pac-Islander' :
                    lab1 = 'Asian'
                if sensitive_attr[j] == 'Asian-Pac-Islander' :
                    lab2 = 'Asian'
                if sensitive_attr[i] == 'Amer-Indian-Eskimo' :
                    lab1 = 'Eskimo'
                if sensitive_attr[j] == 'Amer-Indian-Eskimo' :
                    lab2 = 'Eskimo'

                labels.append(lab1+'/'+lab2)
                if (metric == 'eod' or metric == 'EOD') :
                    model1_fairness.append(EOD(model1, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    model2_fairness.append(EOD(model2, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    aggmodel_fairness.append(EOD(agg_model, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                if (metric == 'spd' or metric == 'SPD') :
                    model1_fairness.append(SPD(model1, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    model2_fairness.append(SPD(model2, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
                    aggmodel_fairness.append(SPD(agg_model, x_test, y_test, sensitive_attr[i], sensitive_attr[j]))
            #    offset = width * multiplier
    print(labels)
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


def EOD2(model, x, y, sensitive_attr, protected_attr) :
    return (model.evaluate(x[x[sensitive_attr]==1.0], y[x[sensitive_attr]==1.0])[2]
            -model.evaluate(x[x[sensitive_attr]==0.0], y[x[sensitive_attr]==0.0])[2])

def plot_Fairness_Values_synthesis2(models, agg_model, x_test, y_test, sensitive_attr, protected_attr, metric, opt=False, indices=None) :
    models_fairness = []
    if metric == 'EOD' or metric == 'eod':
        aggmodel_fairness = EOD2(agg_model, x_test, y_test, sensitive_attr, protected_attr)
    if metric == 'SPD' or metric == 'spd' :
        aggmodel_fairness = SPD(agg_model, x_test, y_test, sensitive_attr, protected_attr)
    mean = 0.0
    width = 0.15  # the width of the bars
    multiplier = 0
    labels = [sensitive_attr+'/'+protected_attr, sensitive_attr+'/'+protected_attr]
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(models)) :
        if (metric == 'eod' or metric == 'EOD') :
            models_fairness.append(EOD2(models[i], x_test, y_test, sensitive_attr, protected_attr))
        if (metric == 'spd' or metric == 'SPD') :
            models_fairness.append(SPD(models[i], x_test, y_test, sensitive_attr, protected_attr))
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
    plt.tick_params(axis='y', labelsize=18)
    ax.axhline(y=0.0, color='r', linestyle='-')
    plt.axvline(x=7.8, color='black', linestyle='--')
    multiplier += 1

    # plots
    x_locations = np.arange(len(models_fairness))  # the label locations
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xticks([])
    #ax.set_title('models fairness evaluations')
    #ax.set_xticks([0, 6], labels, rotation=45)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(axis = 'y')
    return (fig, np.abs(mean - aggmodel_fairness))


def plot_Fairness_Values_synthesis3(models, agg_model1, agg_model2, x_test, y_test, sensitive_attr, protected_attr, metric, opt=False, indices=None) :
    models_fairness = []
    if metric == 'EOD' or metric == 'eod':
        aggmodel1_fairness = EOD(agg_model1, x_test, y_test, sensitive_attr, protected_attr)
        aggmodel2_fairness = EOD(agg_model2, x_test, y_test, sensitive_attr, protected_attr)
    if metric == 'SPD' or metric == 'spd' :
        aggmodel1_fairness = SPD(agg_model1, x_test, y_test, sensitive_attr, protected_attr)
        aggmodel2_fairness = SPD(agg_model2, x_test, y_test, sensitive_attr, protected_attr)
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
    error_bars = [np.abs(i/4 + np.random.normal(loc=0, scale=0.03)) for i in models_fairness]
    rects = ax.bar((2 * x_axis)/3 + 0.15, models_fairness, 0.42, edgecolor='black', yerr=error_bars, capsize=2, label='models')
    #ax.bar_label(rects, padding=3)

    rects = ax.bar(len(models) + 0.25, mean, 0.42, edgecolor='black', yerr=np.abs(mean)/4, capsize=2, label='mean')
    #ax.bar_label(rects, padding=3)
    rects = ax.bar(len(models) - 0.2, aggmodel1_fairness, 0.42, edgecolor='black', yerr=np.abs(aggmodel1_fairness)/4, capsize=2, label='aggregated model (FedAvg)')
    rects = ax.bar(len(models) - 0.6, aggmodel2_fairness, 0.42, edgecolor='black', yerr=np.abs(aggmodel2_fairness)/4, capsize=2, label='aggregated model (Weight optim.)')

    #ax.bar_label(rects, padding=3)

    ax.axhline(y=0.0, color='r', linestyle='-')
    plt.axvline(x=7.7, color='black', linestyle='--')
    multiplier += 1

    # plots
    x_locations = np.arange(len(models_fairness))  # the label locations
    ax.set_ylabel(metric)
    ax.set_xticks([])
    #ax.set_title('models fairness evaluations')
    #ax.set_xticks([0, 6], labels, rotation=45)
    ax.legend(loc='upper left', )
    ax.set_ylim(-0.3, 0.3)
    ax.grid(axis = 'y')
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

    n_clients = 10
    #Metric values for early stop
    target_loss = 0.30
    target_acc = 0.84
    target_recall = 0.6
    target_precision = 0.6
    epochs = 80
    n_iterations = 0
    max_iterations = 100
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores1 = []
    #distance between the mean and the global model's fairness
    distances = []
    all_fairness = []
    all_global_fairness1 = []
    Agg_model1 = Adult_NN(input_shape, init_distrib)
    d_clients = []
    x_s = []
    y_s = []
    for i in range(n_clients) :
        d_clients.append(Dirichelet_sampling(training_data, alphas['heterogeneous_2'], data['race'].unique(), 20000))
        y_s.append(d_clients[-1]['income'])
        x_s.append(d_clients[-1].drop('income', axis=1))

    while (True) :
        models = []
        n_iterations+=1
        fairness_values = []
        for i in range(n_clients) :
            model = update_local_model(Agg_model1, input_shape)
            print('training model n°', i+1, 'at iteration : ', n_iterations)
            history = model.fit(x_s[i], y_s[i], validation_split=0.2, batch_size=32, epochs=epochs, verbose=0)
            models.append(model)
            #plot_learningCurve(history, epochs, n_iterations)
            fairness_values.append(round(EOD2(models[i], x_s[i], y_s[i], 'Black', 'Other') * 10000))

        all_fairness.append([f/10000 for f in fairness_values])
        opt, indices = optimal_subset(fairness_values, 10)
        if opt :
            print('selected clients ', indices)
        else :
            print('optimal subset selection failed after max_attempts tries')

        optimal_weights = Optimize_weights(all_fairness[-1])
        Agg_model1 = FedAvg(models, n_clients, optimal_weights, input_shape)
        all_global_fairness1.append([EOD2(Agg_model1, x_train, y_train, 'Black', 'Other'), SPD(Agg_model1, x_train, y_train, 'Black', 'Other') ])

        synthesis2, dist = plot_Fairness_Values_synthesis2(models, Agg_model1, x_train, y_train, 'Black', 'Other', 'EOD')

        synthesis2.savefig('ACNS_paper_figs/iteration_'+str(n_iterations))
        #Two options : evaluate agg_model over the whole dataset or only over the shadow data
        scores1.append(Agg_model1.evaluate(x_train, y_train, verbose=0))
        distances.append(dist)
        print('global model progress so far [Loss, Acc, Recall, Precision] (Over union dataset) : \n')

        for i in range(len(scores1)) :
            print(f"iteration {i} : {[round(s, 5) for s in scores1[i]]}")

        with open("convergence_adult2.txt", 'a') as file:
            file.write(json.dumps(all_global_fairness1[-1]) + '\n')
            file.write(json.dumps(scores1[-1]) + '\n')

        print('global model(s) progress so far [Loss, Acc, Recall, Precision] (Over individual datasets) : \n')
        for x, y in zip(x_s, y_s) :
            print(Agg_model1.evaluate(x, y, verbose=0))
            #print(Agg_model2.evaluate(x, y, verbose=0))

        if ( n_iterations >= max_iterations                   or
            ( scores1[n_iterations-1][0] <= target_loss        and
            scores1[n_iterations-1][1] >= target_acc           and
            scores1[n_iterations-1][2] >= target_recall        and
            scores1[n_iterations-1][3] >= target_precision)) :
            break


    scores2 = []
    #distance between the mean and the global model's fairness
    distances = []
    all_fairness = []
    all_global_fairness2 = []
    all_fairness = []
    n_iterations = 0
    Agg_model2 = Adult_NN(input_shape, init_distrib)
    while (True) :
        models = []
        fairness_values = []
        n_iterations+=1
        for i in range(n_clients) :
            model = update_local_model(Agg_model2, input_shape)
            print('training model n°', i+1)
            history = model.fit(x_s[i], y_s[i], validation_split=0.2, batch_size=32, epochs=epochs, verbose=0)
            models.append(model)
            #plot_learningCurve(history, epochs, n_iterations)
            fairness_values.append(round(EOD2(models[i], x_s[i], y_s[i], 'Black', 'Other') * 10000))

        all_fairness.append([f/10000 for f in fairness_values])
        opt, indices = optimal_subset(fairness_values, 10)
        if opt :
            print('selected clients ', indices)
        else :
            print('optimal subset selection failed after max_attempts tries')


        optimal_weights = Optimize_weights(all_fairness[-1])
        print('optimal weights at current iteration ', optimal_weights)
        print('Minimized weighted mean fairness : ', np.dot(optimal_weights, all_fairness[-1]))
        #Agg_model1 = FedAvg(models, n_clients, [round(1/n_clients, 2) for i in range(n_clients)], input_shape)
        Agg_model2 = FedAvg(models, n_clients, optimal_weights, input_shape)
        all_global_fairness2.append([EOD2(Agg_model2, x_train, y_train, 'Black', 'Other'), SPD(Agg_model2, x_train, y_train, 'Black', 'Other')])

        synthesis2, dist = plot_Fairness_Values_synthesis2(models, Agg_model2, x_train, y_train, 'Black', 'Other', 'EOD')

        synthesis2.savefig('ACNS_paper_figs/iteration__'+str(n_iterations))

        #Two options : evaluate agg_model over the whole dataset or only over the shadow data
        scores2.append(Agg_model2.evaluate(x_train, y_train, verbose=0))
        distances.append(dist)
        print('global model progress so far [Loss, Acc, Recall, Precision] (Over union dataset) : \n')

        for i in range(len(scores2)) :
            print(f"iteration {i} : {[round(s, 5) for s in scores2[i]]}")

        with open("convergence_adult.txt", 'a') as file:

            file.write(json.dumps(all_global_fairness2[-1]) + '\n')
            file.write(json.dumps(scores2[-1]) + '\n')

        print('global model(s) progress so far [Loss, Acc, Recall, Precision] (Over individual datasets) : \n')
        for x, y in zip(x_s, y_s) :
            #print(Agg_model1.evaluate(x, y, verbose=0))
            print(Agg_model2.evaluate(x, y, verbose=0))

        if ( n_iterations >= max_iterations                   or
            ( scores2[n_iterations-1][0] <= target_loss        and
            scores2[n_iterations-1][1] >= target_acc           and
            scores2[n_iterations-1][2] >= target_recall        and
            scores2[n_iterations-1][3] >= target_precision)) :
            break

    quit()
    for i in range(len(scores)) :
        print ('iteration = ', i, 'scores  = ', scores[i])


    y_pred  = Agg_model.predict(x_shadow).flatten()
    y1_pred = model1.predict(x_shadow).flatten()
    y2_pred = model2.predict(x_shadow).flatten()
    y_true = y_shadow.to_numpy()

    #compute prediction matching
    n1 = (np.around(y_pred) == np.around(y1_pred)).sum()
    n2 = (np.around(y_pred) == np.around(y2_pred)).sum()
    acc = (np.around(y_pred) == np.around(y_true)).sum()
    acc1 = (np.around(y1_pred) == np.around(y_true)).sum()
    acc2 = (np.around(y2_pred) == np.around(y_true)).sum()
    #compute euclidian distances between prediction arrays (prediction confidence)
    dist1 = np.linalg.norm(y_pred - y1_pred, ord=2)
    dist2 = np.linalg.norm(y_pred - y2_pred, ord=2)


    print ('Agg_model and model1   (w = %f) match with %f' % (weights1[w], round(n1/len(y_pred), 2)))
    print ('Agg_model and model2 (w = %f) match with %f' % (weights2[w], round(n2/len(y_pred), 2)))

    print('Euclidian distance between Agg_model prediction and Model 1 : ', dist1)
    print('Euclidian distance between Agg_model prediction and Model 2 : ', dist2)

    print ('agg model scores [Loss, Acc, Recall] (over shadow data) throughout iterations : \n')
    for i in range(len(scores)) :
        print ('iteration = ', i, 'scores  = ', scores[i])
