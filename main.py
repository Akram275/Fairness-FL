from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import confusion_matrix
import csv
#Local imports
from fade_utils import *
from fedAvg_utils import *


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

#Weights initialization
random.seed(datetime.now().timestamp())
SEED=random.randint(0, 123494321)
init_distrib = tf.initializers.HeUniform(seed=SEED)





def get_testdata(datasets, min_records):
    sampled_dfs = []
    sampled_labels = []
    for attributes_df, labels_df in datasets:
        # Ensure indices are aligned
        attributes_df = attributes_df.reset_index(drop=True)
        labels_df = labels_df.reset_index(drop=True)

        # Sample min_records from attributes_df
        sampled_attributes_df = attributes_df.sample(n=min_records, random_state=42)

        # Get corresponding labels using the sampled indices
        sampled_labels_df = labels_df.loc[sampled_attributes_df.index]

        sampled_dfs.append(sampled_attributes_df)
        sampled_labels.append(sampled_labels_df)

    # Concatenate all sampled features and labels
    test_features = pd.concat(sampled_dfs, ignore_index=True)
    test_labels = pd.concat(sampled_labels, ignore_index=True)

    return test_features, test_labels



def run_training(task, datasets, epochs, max_iterations, mode, centralized_test) :
    #Sample a test dataset from all clients' datasets to eval the aggregated model at each iteration
    test_features, test_labels = get_testdata(datasets, 1000)

    n_iterations = 0
    n_clients=len(datasets)
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores = []
    #distance between the mean and the global model's fairness
    SPD_global = []
    EOD_global = []
    FPRD_global = []
    FNRD_global = []
    models = []
    input_shape = (datasets[0][0].shape[1],)

    #Do we want to compare against a centralized training baseline
    #(yes --> do it and save the metrics at the first row of csv file)?
    if centralized_test :
        centralized_x = pd.concat([dataset[0] for dataset in datasets])
        centralized_y = pd.concat([dataset[1] for dataset in datasets])
        centralized_model = Adult_NN(input_shape, init_distrib)
        print('Centrally training a model on the union data ...')
        centralized_model.fit(centralized_x, centralized_y, epochs=50, verbose=0)
        print('evaluating the centralized model')
        eval = centralized_model.evaluate(test_features, test_labels)
        conv_file_name = 'FADE_data/'+task+'/'+mode+'/convergence_'+metric+'.csv'
        curr_global_fairness = [float(EOD(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr)),
                                float(SPD(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr)),
                                float(FPRD(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr)),
                                float(FNRD(centralized_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))]

        with open(conv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(eval)

        fairness_file_name = 'FADE_data/'+task+'/'+mode+'/fairness_'+metric+'.csv'
        with open(fairness_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(curr_global_fairness)

        #Free heavy data from memory
        del centralized_x
        del centralized_y
        del centralized_model

    #This NN (For Adult) performs well on FolkTables data
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
            # Reset indices to ensure alignment
            datasets[i][0] = datasets[i][0].reset_index(drop=True)
            datasets[i][1] = datasets[i][1].reset_index(drop=True)

            # Sample 70% for training data
            x_train = datasets[i][0].sample(n=int(datasets[i][0].shape[0] * 0.7), replace=False)

            # Get the corresponding labels for x_train
            y_train = datasets[i][1].loc[x_train.index]

            # Use the remaining 30% for testing data
            x_test = datasets[i][0].drop(x_train.index)  # Use the rest for testing
            y_test = datasets[i][1].loc[x_test.index]    # Get corresponding labels for x_test

            if mode == 'FairFed' : #FairFed requires local-debiasing prior to training
                x_train, y_train, _ = reweighing_debias(x_train, y_train, sensitive_attribute=attr)

            history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epochs, verbose=0)
            models.append(model)
            print(f"model {i} local performance : {model.evaluate(x_test, y_test, verbose=0)}")
            print(metric)
            if metric == 'eod' :
                curr_fairness.append(EOD(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))
            if metric == 'spd' :
                curr_fairness.append(SPD(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))
            if metric == 'fprd' :
                curr_fairness.append(FPRD(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))
            if metric == 'fnrd' :
                curr_fairness.append(FNRD(model, x_test, y_test, attr=attr, protected=protected_attr, privileged=privileged_attr))

        print('models fairness : ', curr_fairness)
        fedavg_weights = [datasets[i][0].shape[0]/sum(ds_sizes) for i in range(len(datasets))]

        if mode == 'FedAvg' :
            Agg_model = FedAvg(models, n_clients, fedavg_weights, input_shape)
        elif mode == 'OptW' :
            optimal_weights = Optimize_weights(curr_fairness)
            print('Optimal weights at current iteration : ', optimal_weights)
            Agg_model = FedAvg(models, n_clients, optimal_weights, input_shape)
        elif mode == 'SSP' :
            scaled_fairness = [(f * w) for (f, w) in zip(curr_fairness, fedavg_weights)]
            max_attempts = 7
            epsilon = 10**(-(max_attempts-1))
            n_attempts = 1
            optimal_sum, optimal_subset = approximate_subset_sum_floats(scaled_fairness, 0.0, epsilon)
            while (((optimal_sum, optimal_subset) == (None, []) or len(optimal_subset) < 3) and n_attempts < max_attempts) :
                n_attempts +=1
                epsilon = 10 * epsilon
                print(f'failed to find optimal subset at attempt {n_attempts} with epsilon {epsilon}')
                optimal_sum, optimal_subset = approximate_subset_sum_floats(scaled_fairness, 0.0, epsilon)
            if optimal_subset != [] :
                print(f'Optimal subset found : {optimal_subset} with sum {optimal_sum}')
                opt_models = [models[i] for i in optimal_subset]
                opt_fedavg_weight = [fedavg_weights[i] for i in optimal_subset]
                #normalize optimal clients' FedAvg weights
                opt_fedavg_weight = [k/sum(opt_fedavg_weight) for k in opt_fedavg_weight]
                Agg_model = FedAvg(opt_models, len(optimal_subset), opt_fedavg_weight, input_shape)
            elif optimal_subset == [] :
                print(f'SSP Failure after {n_attempts} attempts --> FedAvg')
                Agg_model = FedAvg(models, n_clients, fedavg_weights, input_shape)
        elif mode == 'FairFed' :
            Agg_model = FedAvg(models, n_clients, fedavg_weights, input_shape)

        #Evaluate global model on the union validation dataset
        scores.append(Agg_model.evaluate(test_features, test_labels))

        for i in range(len(scores)) :
            print(f"iteration {i} : {[round(s, 5) for s in scores[i]]}")
        SPD_global.append(SPD(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        EOD_global.append(EOD(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        FPRD_global.append(FPRD(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        FNRD_global.append(FNRD(Agg_model, test_features, test_labels, attr=attr, protected=protected_attr, privileged=privileged_attr))
        synthesis2, dist = plot_Fairness_Values_synthesis2(models, Agg_model, test_features, test_labels, metric=metric, attr=attr, protected=protected_attr, privileged=privileged_attr)
        synthesis2.savefig('FADE_data/'+task+'/'+mode+'/iteration_'+str(n_iterations))

        curr_global_fairness = [float(EOD_global[-1]), float(SPD_global[-1]), float(FPRD_global[-1]), float(FNRD_global[-1])]

        conv_file_name = 'FADE_data/'+task+'/'+mode+'/convergence_'+metric+'.csv'
        with open(conv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(scores[-1])

        fairness_file_name = 'FADE_data/'+task+'/'+mode+'/fairness_'+metric+'.csv'
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
    if task not in ['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage', 'Adult'] :
        print('unsported task (not in : ACSIncome, ACSEmployment, ACSPublicCoverage)')
        sys.exit(1)
    metric = sys.argv[2]
    if metric not in ['spd', 'eod', 'fnrd', 'fprd'] :
        print('unsported metric (not in : spd, eod, fnrd, fprd)')
        sys.exit(1)

    if task == 'ACSIncome' :
        if metric == 'eod' or metric == 'fnrd':
            states = income_eod_states
        if metric== 'spd' or metric == 'fprd':
            states = income_spd_states
    if task == 'ACSEmployment' :
        if metric == 'spd' or metric == 'fprd':
            states = employment_spd_states
        if metric == 'eod' or metric == 'fnrd':
            states = employment_eod_states
    if task == 'ACSPublicCoverage' :
        if metric == 'spd' or metric == 'fprd':
            states = publiccoverage_spd_states
        if metric == 'eod' or metric == 'fnrd':
            states = publiccoverage_eod_states

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    datasets = []

    #Some clusters of 10 states to simulate clients data
    if task in ['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage'] :
        attr='SEX'
        protected_attr=1.0
        privileged_attr=2.0
        for state in states :
            print('Client simulates '+state+' ACS data')
            acs_data = data_source.get_data(states=[state], download=True)
            if task == 'ACSEmployment' :
                datasets.append([ACSEmployment.df_to_pandas(acs_data)[0], ACSEmployment.df_to_pandas(acs_data)[1]])
            if task == 'ACSPublicCoverage' :
                datasets.append([ACSPublicCoverage.df_to_pandas(acs_data)[0], ACSPublicCoverage.df_to_pandas(acs_data)[1]])
            if task == 'ACSIncome' :
                datasets.append([ACSIncome.df_to_pandas(acs_data)[0], ACSIncome.df_to_pandas(acs_data)[1]])
            print('dataset size : ', datasets[-1][0].shape[0])

    elif task == 'Adult' :
        attr='sex'
        protected_attr=1.0
        privileged_attr=0.0
        data = pd.read_csv('datasets/adult.csv')
        data.replace('?', np.NaN)
        print("Data preprocessing ...")
        pre_processed_data = data_PreProcess(data)
        alphas = create_alphas(len(data['race'].unique()))
        for i in range(10) :
            #Introduce some heterogeneity level alpha = 0.5
            d_clients = Dirichelet_sampling(pre_processed_data, alphas['heterogeneous_2'], data['race'].unique(), 20000)
            y = d_clients['income']
            x = d_clients.drop('income', axis=1)
            datasets.append([x, y])


    #Perform a first centralized + FedAvg baseline training
    run_training(task, datasets, epochs=1, max_iterations=150, mode='FedAvg', centralized_test=True)
    run_training(task, datasets, epochs=1, max_iterations=150, mode='FairFed', centralized_test=False)
    #Perform a second with our FADE solutions
    run_training(task, datasets, epochs=1, max_iterations=150, mode='OptW', centralized_test=False)
    run_training(task, datasets, epochs=1, max_iterations=150, mode='SSP', centralized_test=False)
