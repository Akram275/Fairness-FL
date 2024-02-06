import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
import sys
import random
import pandas as pd
import numpy as np
import math

import array
import shap
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from scipy.optimize import minimize
from scipy.optimize import linprog
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def FairFace_CNN(lr=0.001) :
    # Create a simple CNN model
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss='binary_crossentropy',
              metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                         tf.keras.metrics.Precision(name='Precision'),
                         tf.keras.metrics.Recall(name='recall')
                         ])

    return model


def generator_subpopulation(df, attr, val) :
    """
    Generate data_flow from df with given criteria (df[attr]==val)
    """
    generator_subpopulation = val_datagen.flow_from_dataframe(
            df[df[attr]==val],
            directory='',
            x_col='file',
            y_col='gender',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary')

    return generator_subpopulation

def EOD(model, df, attr, protected, privileged) :
    return model.evaluate(generator_subpopulation(df, attr, protected), verbose=0)[3] - model.evaluate(generator_subpopulation(df, attr, privileged), verbose=0)[3]

def SPD(model, df, attr, protected, privileged) :
    return np.round(model.predict(generator_subpopulation(df, attr, protected), verbose=0)).mean() - np.round(model.predict(generator_subpopulation(df, attr, privileged), verbose=0)).mean()

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
        'extremely heterogeneous' : [1/1000     for i in range(dim)] #inutilisable -> Des valeurs trop petite (2^-128 => np.NaN)
    }
    return alphas

def Dirichelet_sampling(data, alphas, values,  n_lines) :
    #values = data[col].unique()
    assert (n_lines <= len(data.axes[0]))
    assert (len(values) == len(alphas))
    #sample a distribution from dir(alphas)
    s = np.random.dirichlet(tuple(alphas), 1).tolist()[0]
    for i in range(len(values)) :
        print (values[i], '  :  ', np.round(s[i] * n_lines), ' samples')
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
                groups.append(data[data['race']==values[i]].sample(n=round(n_lines * s[i]), replace=True))
            else :
                groups.append(data[data['race']==values[i]].sample(n=1))

    return pd.concat(groups)



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
        layer_mean = tf.math.reduce_sum([tf.convert_to_tensor(grad_list_tuple[0]), tf.convert_to_tensor(grad_list_tuple[1])] , axis=0)
        avg_grad.append(layer_mean)

    return avg_grad

#Two versions :
#.numpy() to convert a tensor to a numpy array doesnt work when eager_exection is disabled
#repace with .eval() after creating a tensorflow Session()

def FedAvg_eager_exec_disabled(models, n, clients_weights, input_shape) :
    scaled_weights = []

    global_model = FairFace_CNN()
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
    global_model = FairFace_CNN()
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))
    avg_weights = sum_scaled_weights(scaled_weights)
    global_model.set_weights([avg_weight_layer.numpy() for avg_weight_layer in avg_weights])
    #global_model.set_weights(avg_weights)
    return global_model


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


if __name__ == '__main__':

    train_csv_path = 'train_labels.csv'
    val_csv_path = 'val_labels.csv'

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Encode gender labels
    label_encoder = LabelEncoder()
    train_df['gender'] = label_encoder.fit_transform(train_df['gender']).astype(str)
    val_df['gender'] = label_encoder.transform(val_df['gender']).astype(str)

    #Validation data identical for all.
    batch_size = 32
    image_size = (128, 128)  # FairFace image size
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True) #Not sure for the horizontal_flip

    val_datagen = ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory='',
        x_col='file',
        y_col='gender',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # Create a list of dataframes
    n_clients = 2
    n_iterations = 2
    local_epochs = 10
    alphas = create_alphas(len (train_df['race'].unique()))
    datasets = []
    for i in range(n_clients) :
        #Sample from a Dirichlet distribution 10k samples for training
        datasets.append(Dirichelet_sampling(train_df, alphas['heterogeneous_10'], train_df['race'].unique(), 5000))

    #init global model
    global_model = FairFace_CNN()

    for t in range(n_iterations) :
        models = []
        eods = []
        for i in range(n_clients) :
            train_generator = train_datagen.flow_from_dataframe(
                datasets[i],
                directory='',
                x_col='file',
                y_col='gender',
                target_size=image_size,
                batch_size=batch_size,
                class_mode='binary'
                )
            model = update_local_model(global_model, (128, 128, 3))

            history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=local_epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size)

            models.append(model)
            eods.append(EOD(models[i], val_df, 'race', 'Black', 'White'))
            print(eods[-1])

        for gr1 in val_df['race'].unique() :
            for gr2 in val_df['race'].unique() :
                if gr1 != gr2 :
                    print('model 1 SPD ', SPD(models[0], val_df, 'race', gr1, gr2))
                    print('model 2 SPD ', SPD(models[1], val_df, 'race', gr1, gr2))
                    print('global EOD ', SPD(global_model, val_df, 'race', gr1, gr2))
                    print('\n\n\n')
            #plot_learningCurve(history, str(0), str(0))
        optimal_weights = Optimize_weights(eods)
        #Change for optimal weights below
        global_model = FedAvg(models, n_clients, [1/n_clients for i in range(n_clients)], (128, 128, 3))
