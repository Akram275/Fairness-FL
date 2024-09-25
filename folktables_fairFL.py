from n_clients_improoved_FL import *
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage
from SSP_tests import *
import csv
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering

#Optimal states (with most opposing unfairness)
income_eod_states = ['AL', 'AK', 'AR', 'DE', 'FL', 'ME', 'MD', 'NE', 'NY', 'WA']
income_spd_states =  ['AL', 'AK', 'AZ', 'GA', 'ID', 'MS', 'MT', 'NV', 'SD', 'WV']

employment_spd_states = ['AL', 'AK', 'AZ', 'CO', 'DE', 'NE', 'PA', 'WA', 'WI', 'WY']
employment_eod_states = ['AL', 'AK', 'AZ', 'CA', 'MS', 'NE', 'NJ', 'NY', 'OR', 'TX']

publiccoverage_spd_states = ['AL', 'AK', 'AZ', 'CT', 'NE', 'ND', 'OK', 'PA', 'SD', 'TX']
publiccoverage_eod_states = ['AL', 'AK', 'AZ', 'DE', 'KS', 'MO', 'NV', 'OK', 'SC', 'WI']

all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

#Pick groups for which the fairness analysis will be carried. Full details on attributes and groups in
#https://arxiv.org/pdf/2108.04884

attr='SEX'
protected_attr=1.0
privileged_attr=2.0



#Weights initialization
random.seed(datetime.now().timestamp())
SEED=random.randint(0, 123494321)
init_distrib = tf.initializers.HeUniform(seed=SEED)

def SPD2(model, x, y, attr, protected=None, privileged=None) :
    if attr == 'SEX'  :
        return np.mean(model.predict(x[x['SEX']==1.0])) - np.mean(model.predict(x[x['SEX']==2.0]))

    return np.mean(model.predict(x[x[attr]==protected])) - np.mean(model.predict(x[x[attr]==privileged]))

def EOD2(model, x, y, attr, protected=None, privileged=None) :
    #Difference in Recall
    if attr == 'SEX' :
            return (model.evaluate(x[x['SEX']==1.0], y[x['SEX']==1.0],verbose=0)[2]
            - model.evaluate(x[x['SEX']==2.0], y[x['SEX']==2.0],verbose=0)[2])

    return (model.evaluate(x[x[attr]==protected], y[x[attr]==protected], verbose=0)[2]
    - model.evaluate(x[x[attr]==privileged], y[x[attr]==privileged], verbose=0)[2])

def plot_Fairness_Values_synthesis2(models, agg_model, x_test, y_test, metric, opt=False, indices=None) :
    models_fairness = []
    if metric == 'EOD' or metric == 'eod':
        aggmodel_fairness = EOD2(agg_model, x_test, y_test, attr, privileged_attr, protected_attr)
    if metric == 'SPD' or metric == 'spd' :
        aggmodel_fairness = SPD2(agg_model, x_test, y_test, attr, privileged_attr, protected_attr)
    mean = 0.0
    width = 0.15  # the width of the bars
    multiplier = 0
    #labels = [sensitive_attr+'/'+protected_attr, sensitive_attr+'/'+protected_attr]
    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(models)) :
        if (metric == 'eod' or metric == 'EOD') :
            models_fairness.append(EOD2(models[i], x_test, y_test, attr, privileged_attr, protected_attr))
        if (metric == 'spd' or metric == 'SPD') :
            models_fairness.append(SPD2(models[i], x_test, y_test, attr, privileged_attr, protected_attr))
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
    ax.set_xticks([i/10 for i in range(len(states))], states, rotation=0)
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xticks([])
    #ax.set_title('models fairness evaluations')
    #ax.set_xticks([0, 6], labels, rotation=45)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(axis = 'y')
    return (fig, np.abs(mean - aggmodel_fairness))


def get_testdata(datasets, min_records) :
    sampled_dfs = []
    sampled_labels = []
    for attributes_df, labels_df, _ in datasets:
        sampled_attributes_df = attributes_df.sample(n=min_records, random_state=42)  # Sample attributes
        sampled_labels_df = labels_df.loc[sampled_attributes_df.index]                # Align sampled labels
        sampled_dfs.append(sampled_attributes_df)
        sampled_labels.append(sampled_labels_df)

    test_features = pd.concat(sampled_dfs, ignore_index=True)
    test_labels = pd.concat(sampled_labels, ignore_index=True)
    return test_features, test_labels


def run_training(task, datasets, epochs, max_iterations, mode, centralized_test) :
    #Sample a test dataset from all clients' datasets to eval the aggregated model at each iteration
    test_features, test_labels = get_testdata(datasets, 1000)

    #Metric values for early stop
    target_loss = 0.30
    target_acc = 0.84
    target_recall = 0.6
    target_precision = 0.6
    n_iterations = 0
    n_clients=len(datasets)
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores = []
    #distance between the mean and the global model's fairness
    distances = []
    SPD_global = []
    EOD_global = []
    all_global_fairness1 = []
    models = []
    input_shape = (datasets[0][0].shape[1],)

    #Do we want to compare against a centralized training baseline
    #(yes --> do it and save the metrics at the first row of csv file)?
    if centralized_test :
        centralized_x = pd.concat([dataset[0] for dataset in datasets])
        centralized_y = pd.concat([dataset[1] for dataset in datasets])
        centralized_model = Adult_NN(input_shape, init_distrib)
        print('training a centralied model on the union data ...')
        centralized_model.fit(centralized_x, centralized_y, epochs=50, verbose=0)
        print('evaluating the centralized model')
        eval = centralized_model.evaluate(test_features, test_labels)
        conv_file_name = 'CIKM_data/'+task+'/'+mode+'/convergence.csv'
        curr_global_fairness = [float(EOD2(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr)),
                                float(SPD2(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))]

        with open(conv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(eval)

        fairness_file_name = 'CIKM_data/'+task+'/'+mode+'/fairness.csv'
        with open(fairness_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(curr_global_fairness)
        del centralized_x
        del centralized_y
        del centralized_model


    Agg_model = Adult_NN(input_shape, init_distrib)
    ds_sizes = [datasets[i][0].shape[0] for i in range(len(datasets))]
    while (True) :
        curr_fairness = []
        models = []
        n_iterations+=1
        fairness_values = []
        for i in range(n_clients) :
            model = update_local_model(Agg_model, input_shape)
            print('training model n°', i+1, 'at iteration : ', n_iterations)
            x_train = datasets[i][0].sample(n=int(datasets[i][0].shape[0] * 0.7), replace=False)
            y_train = datasets[i][1].loc[x_train.index]
            x_test = datasets[i][0].drop(x_train.index) #Use the rest for test
            y_test = datasets[i][1].loc[x_test.index]
            history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epochs, verbose=0)
            models.append(model)
            print(f"model {i} local performance : {model.evaluate(x_test, y_test, verbose=0)}")
            print(metric)
            if metric == 'eod' :
                curr_fairness.append(EOD2(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))
            if metric == 'spd' :
                curr_fairness.append(SPD2(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))
        print('models fairness : ', curr_fairness)
        fedavg_weights = [datasets[i][0].shape[0]/sum(ds_sizes) for i in range(len(datasets))]

        if mode == 'FedAvg' :
            Agg_model = FedAvg(models, n_clients, fedavg_weights, input_shape)
        elif mode == 'WO' :
            optimal_weights = Optimize_weights(curr_fairness)
            print('Optimal weights at current iteration : ', optimal_weights)
            Agg_model = FedAvg(models, n_clients, optimal_weights, input_shape)
        elif mode == 'SSP' :
            scaled_fairness = [(f * w) for (f, w) in zip(curr_fairness, fedavg_weights)]
            max_attempts = 7
            epsilon = 10**(-(max_attempts-1))
            n_attempts = 1
            optimal_sum, optimal_subset = approximate_subset_sum_floats(scaled_fairness, 0.0, epsilon)
            while ((optimal_sum, optimal_subset) == (None, []) or len(optimal_subset) < 3 or n_attempts < max_attempts) :
                n_attempts +=1
                epsilon = 10 * epsilon
                print(f'failed to find optimal subset at attempt {n_attempts} with epsilon {epsilon}')
                optimal_sum, optimal_subset = approximate_subset_sum_floats(scaled_fairness, 0.0, epsilon)
            if optimal_sum != [] :
                print(f'Optimal subset found : {optimal_subset} with sum {optimal_sum}')
                opt_models = [models[i] for i in optimal_subset]
                opt_fedavg_weight = [fedavg_weights[i] for i in optimal_subset]
                #normalize optimal clients' FedAvg weights
                opt_fedavg_weight = [k/sum(opt_fedavg_weight) for k in opt_fedavg_weight]
                Agg_model = FedAvg(opt_models, len(optimal_subset), opt_fedavg_weight, input_shape)
            elif optimal_sum == None :
                print('SSP Failure --> FedAvg')
                Agg_model = FedAvg(models, n_clients, optimal_weights, input_shape)
        #Evaluate global model on the union validation dataset
        scores.append(Agg_model.evaluate(test_features, test_labels))

        for i in range(len(scores)) :
            print(f"iteration {i} : {[round(s, 5) for s in scores[i]]}")
        SPD_global.append(SPD2(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        synthesis2, dist = plot_Fairness_Values_synthesis2(models, Agg_model, test_features, test_labels, metric=metric)
        synthesis2.savefig('CIKM_data/'+task+'/'+mode+'/iteration_'+str(n_iterations))
        EOD_global.append(EOD2(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        curr_global_fairness = [float(EOD_global[-1]), float(SPD_global[-1])]

        conv_file_name = 'CIKM_data/'+task+'/'+mode+'/convergence.csv'
        with open(conv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(scores[-1])

        fairness_file_name = 'CIKM_data/'+task+'/'+mode+'/fairness.csv'
        with open(fairness_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(curr_global_fairness)

        if n_iterations == max_iterations :
            break


if __name__ =='__main__' :
    if len(sys.argv) < 3:
        print("Usage: python3 folkTables_FairFL.py \"task\" (ASCEmployement, ASCIncome, ASCPublicCoverage)" )
        sys.exit(1)

    task = sys.argv[1]
    if task not in ['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage'] :
        print('unsported task (not in : ACSIncome, ACSEmployment, ACSPublicCoverage)')
        sys.exit(1)
    metric = sys.argv[2]

    if task == 'ACSIncome' :
        if metric == 'eod' :
            states = income_eod_states
        if metric== 'spd' :
            states = income_spd_states
    if task == 'ACSEmployment' :
        if metric == 'spd' :
            states = employment_spd_states
        if metric == 'eod' :
            states = employment_eod_states
    if task == 'ACSPublicCoverage' :
        if metric == 'spd' :
            states = publiccoverage_spd_states
        if metric == 'eod' :
            states = publiccoverage_eod_states

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    datasets = []

    #Some clusters of 10 states to simulate clients data
    for state in states :
        print('Client simulates '+state+' ACS data')
        acs_data = data_source.get_data(states=[state], download=True)
        if task == 'ACSEmployment' :
            datasets.append(ACSEmployment.df_to_pandas(acs_data))
        if task == 'ACSPublicCoverage' :
            datasets.append(ACSPublicCoverage.df_to_pandas(acs_data))
        if task == 'ACSIncome' :
            datasets.append(ACSIncome.df_to_pandas(acs_data))
        print('dataset size : ', datasets[-1][0].shape[0])

    #Do a first centralized + FedAvg baseline training
    run_training(task, datasets, epochs=1, max_iterations=150, mode='FedAvg', centralized_test=True)
    #Do a second with our FADE solutions
    run_training(task, datasets, epochs=1, max_iterations=150, mode='WO', centralized_test=False)
    run_training(task, datasets, epochs=1, max_iterations=150, mode='SSP', centralized_test=False)
