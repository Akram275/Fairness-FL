from fade_utils import *
from adult_utils import *
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage
from fairness_utils import *
from tensorflow.keras.models import clone_model
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt



def mean_based_eod(model, X_test, y_test, attr, protected_attr, privileged_attr) :
    if attr == 'SEX' :
            return (np.mean(model.predict(X_test[ (X_test['SEX']==1.0) & (y_test == 1)], verbose=0))
            - np.mean(model.predict( X_test[(X_test['SEX']==0.0) & (y_test == 1) ], verbose=0)))

    return (np.mean(model.predict(X_test[( X_test[attr]==protected_attr) & (y_test == 1)], verbose=0))
    - np.mean(model.predict(X_test[ (X_test[attr]==privileged_attr) & (y_test == 1)], verbose=0)))

def mean_based_spd(model, X_test, y_test, attr, protected_attr, privileged_attr) :
    if attr == 'SEX' :
            return (np.mean(model.predict(X_test[X_test['SEX']==1.0], verbose=0))
            - np.mean(model.predict(X_test[X_test['SEX']==0.0], verbose=0)))

    return (np.mean(model.predict(X_test[X_test[attr]==protected_attr], verbose=0))
    - np.mean(model.predict(X_test[X_test[attr]==privileged_attr], verbose=0)))


def Adult_NN(input_shape) :
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED))(inputs)
    BatchNormalization(),
    x = tf.keras.layers.Dense(8, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED))(x)
    BatchNormalization(),
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



def folktables_DNN(input_shape, init_distrib):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape, bias_initializer=init_distrib, kernel_initializer=init_distrib),
        Dense(32, activation='relu', bias_initializer=init_distrib, kernel_initializer=init_distrib),
        Dense(16, activation='relu', bias_initializer=init_distrib, kernel_initializer=init_distrib),
        Dense(8, activation='relu', bias_initializer=init_distrib, kernel_initializer=init_distrib),
        Dense(1, activation='sigmoid', bias_initializer=init_distrib, kernel_initializer=init_distrib)  # Binary classification
    ])
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Precision(name='Precision')

    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=metrics)
    return model


def evaluate_perturbations(model, X_test1, y_test1, X_test2, y_test2, num_samples=100, distance_metric="cosine"):
    """
    Apply multiple perturbations to the model and evaluate fairness smoothness.

    Parameters:
        model: A Keras model to perturb.
        X_test: Test features.
        y_test: Test labels.
        perturb_model_fn: A function that perturbs the model's weights.
        eod_fn: A function that computes the EOD of a model given test data.
        num_samples (int): Number of perturbations to apply.
        distance_metric (str): Metric to compute distance in parameter space.

    Returns:
        list of tuples: [(param_distance, eod_difference), ...]
    """
    results = []


    all_x = pd.concat([X_test1, X_test2])
    all_y = pd.concat([y_test1, y_test2])


    # Compute the EOD for the original model
    original_eod = (mean_based_eod(model, X_test1, y_test1, attr, protected_attr, privileged_attr),
                    mean_based_eod(model, all_x, all_y, attr, protected_attr, privileged_attr),
                    mean_based_eod(model, X_test2, y_test2, attr, protected_attr, privileged_attr))
    original_spd = (mean_based_spd(model, X_test1, y_test1, attr, protected_attr, privileged_attr),
                    mean_based_spd(model, all_x, all_y, attr, protected_attr, privileged_attr),
                    mean_based_spd(model, X_test2, y_test2, attr, protected_attr, privileged_attr))

    original_weights = np.concatenate([w.flatten() for w in model.get_weights()])

    P = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    noisy_model = personalized_model(model, noise_std=1, P=0.0)
    results.append([0.0, original_eod, original_spd])
    for i in range(num_samples):
        # Perturb the model

        noisy_model = personalized_model(noisy_model, noise_std=0.1, P=P[5], straight_line=True)

        # Compute the parameter-space distance
        noisy_weights = np.concatenate([w.flatten() for w in noisy_model.get_weights()])
        if distance_metric == "euclidean":
            param_distance = np.linalg.norm(original_weights - noisy_weights)
        elif distance_metric == "cosine":
            cosine_similarity = np.dot(original_weights, noisy_weights) / (
                np.linalg.norm(original_weights) * np.linalg.norm(noisy_weights)
            )
            param_distance = 1 - cosine_similarity
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        noisy_eod = (mean_based_eod(noisy_model, X_test1, y_test1, attr, protected_attr, privileged_attr),
                    mean_based_eod(noisy_model, all_x, all_y, attr, protected_attr, privileged_attr),
                    mean_based_eod(noisy_model, X_test2, y_test2, attr, protected_attr, privileged_attr))


        noisy_spd = (mean_based_spd(noisy_model, X_test1, y_test1, attr, protected_attr, privileged_attr),
                    mean_based_spd(noisy_model, all_x, all_y, attr, protected_attr, privileged_attr),
                    mean_based_spd(noisy_model, X_test2, y_test2, attr, protected_attr, privileged_attr))

        #print((param_distance, noisy_eod, noisy_spd), '  ', noisy_model.evaluate(X_test1, y_test1, verbose=0))
        print((param_distance, noisy_eod, noisy_spd))
        results.append((param_distance, noisy_eod, noisy_spd))

    return results

def plot_fairness_cloud(results, metric, output_file='fairness_cloud.png'):
    """
    Plot a cloud of points representing the relationship between model weight distance
    and fairness (EOD) in the parameter space using a list of results.

    Parameters:
        results (list): A list where each entry is a tuple or list:
            - results[i][0]: Distance in parameter space.
            - results[i][1]: EOD (fairness) value corresponding to the distance.
        output_file (str): File path to save the plot.

    Returns:
        None
    """
    distances, eod_values, spd_values = zip(*results)

    distances = np.array(distances)
    eod_values = np.array(eod_values)

    # Plot the results
    plt.figure(figsize=(8, 6))
    if metric == 'EOD' :
        plt.scatter(distances, [i[0] for i in eod_values], alpha=0.7, color='blue', linestyle='-', marker='^', edgecolor='black', label=r'EOD$_{D_1}(\theta)$')
        plt.scatter(distances, [i[1] for i in eod_values], alpha=0.7, color='cyan', linestyle='-', marker='^', edgecolor='black', label=r'EOD$_{(D_1 \cup D_2)}(\theta)$')
        plt.scatter(distances, [i[2] for i in eod_values], alpha=0.7, color='green', linestyle='-', marker='^', edgecolor='black', label=r'EOD$_{D_2}(\theta)$')

    if metric == 'SPD' :
        plt.scatter(distances, [i[0] for i in spd_values], alpha=0.7, color='blue', linestyle='-', marker='s', edgecolor='black', label=r'SPD$_{D_1}(\theta)$')
        plt.scatter(distances, [i[1] for i in spd_values], alpha=0.7, color='cyan', linestyle='-', marker='s', edgecolor='black', label=r'SPD$_{(D_1 \cup D_2)}(\theta)$')
        plt.scatter(distances, [i[2] for i in spd_values], alpha=0.7, color='green', linestyle='-', marker='s', edgecolor='black', label=r'SPD$_{D_2}(\theta)$')

    plt.axhline(0, linestyle='--', color='red')
    plt.legend(fontsize=18)
    plt.ylim(-0.30, 0.30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_file}")




def personalized_model(model, noise_std=0.1, P=1.0, lr=0.0001, straight_line=False):

    """
    Clone a Keras model and add noise to its weights scaled by P.

    Parameters:
        model (tf.keras.Model): The original model to be cloned.
        noise_scale (float): The standard deviation of the noise.
        P (float): Scaling factor for the noise in [0, 1].
        straight_line : (bool) Whether our movement follows a straight line path over the param space.
        This is ensured by setting a seed value which makes the step noise value constant
    Returns:
        tf.keras.Model: A new model with noisy weights.
    """
    if straight_line :
        np.random.seed(42)
    # Clone the model structure
    noisy_model = clone_model(model)
    noisy_model.build(model.input_shape)

    # Add noise to the weights of the cloned model
    for original_layer, noisy_layer in zip(model.layers, noisy_model.layers):
        if hasattr(original_layer, 'weights'):
            original_weights = original_layer.get_weights()
            # Add noise to each weight
            noisy_weights = [
                w + (P * np.random.normal(0.0, noise_std, w.shape)) for w in original_weights
            ]
            noisy_layer.set_weights(noisy_weights)

    noisy_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                          tf.keras.metrics.Recall(name='Recall'),
                          tf.keras.metrics.Precision(name='Precision')])
    return noisy_model




task = "Adult"



if task == "Adult" :
    attr='sex'
    protected_attr=1.0
    privileged_attr=0.0
    data = pd.read_csv('datasets/adult.csv')
    data.replace('?', np.NaN)
    alphas = create_alphas(len(data['sex'].unique()))
    print("Data preprocessing ...")
    pre_processed_data = data_PreProcess(data)
    d1 = Dirichelet_sampling(pre_processed_data, alphas['heterogeneous_10'], data['sex'].unique(), 3000)
    d2 = Dirichelet_sampling(pre_processed_data, alphas['heterogeneous_10'], data['sex'].unique(), 3000)
    y_test1 = d1['income']
    X_test1 = d1.drop('income', axis=1)
    y_test2 = pre_processed_data['income']
    X_test2 = pre_processed_data.drop('income', axis=1)



else :
    attr='SEX'
    protected_attr=1.0
    privileged_attr=2.0
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=['WY'], download=True)
    X_test1, y_test1, _ = ACSIncome.df_to_pandas(acs_data)
    acs_data = data_source.get_data(states=['AL'], download=True)
    X_test2, y_test2, _ = ACSIncome.df_to_pandas(acs_data)


input_shape = (X_test1.shape[1],)
model = Adult_NN(input_shape)


distance = "euclidean"

model.fit(X_test1, y_test1, epochs=1)
res = evaluate_perturbations(model, X_test1, y_test1, X_test2, y_test2, distance_metric=distance)
plot_fairness_cloud(res, 'SPD', output_file='fairness_cloud_1_SPD.png')
plot_fairness_cloud(res, 'EOD', output_file='fairness_cloud_1_EOD.png')



model.fit(X_test2, y_test2, epochs=10)
res = evaluate_perturbations(model, X_test1, y_test1, X_test2, y_test2, distance_metric=distance)
plot_fairness_cloud(res, 'SPD', output_file='fairness_cloud_10_SPD.png')
plot_fairness_cloud(res, 'EOD', output_file='fairness_cloud_10_EOD.png')


model.fit(X_test2, y_test2, epochs=40)
res = evaluate_perturbations(model, X_test1, y_test1, X_test2, y_test2, distance_metric=distance)
plot_fairness_cloud(res, 'SPD', output_file='fairness_cloud_50_SPD.png')
plot_fairness_cloud(res, 'EOD', output_file='fairness_cloud_50_EOD.png')


model.fit(X_test1, y_test1, epochs=30)
res = evaluate_perturbations(model, X_test1, y_test1, X_test2, y_test2, distance_metric=distance)
plot_fairness_cloud(res, 'SPD', output_file='fairness_cloud_100_SPD.png')
plot_fairness_cloud(res, 'EOD', output_file='fairness_cloud_100_EOD.png')
