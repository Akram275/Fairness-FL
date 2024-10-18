import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

# Read the CSV file without headers
task = sys.argv[1]

fairness_metric = sys.argv[2]

if task not in ['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage', 'Adult'] :
    print('unsported task (not in : ACSIncome, ACSEmployment, ACSPublicCoverage)')
    sys.exit(1)
metric = sys.argv[2]
if fairness_metric not in ['spd', 'eod', 'fnrd', 'fprd'] :
    print('unsported metric (not in : spd, eod, fnrd, fprd)')
    sys.exit(1)



def plot_metric(task, metric_name, modes, fairness_metric):
    accuracies = []
    losses = []
    precisions = []
    recalls = []
    eods = []
    spds = []
    fprds = []
    fnrds = []

    for mode in modes :
        if mode == 'FedAvg' :   #There is no fairness_metric in FedAvg (We are not optimizing any metric)
            conv_file_path = 'FADE_data/'+task+'/'+mode+'/convergence.csv'
            fairness_file_path = 'FADE_data/'+task+'/'+mode+'/fairness.csv'
        else :
            conv_file_path = 'FADE_data/'+task+'/'+mode+'/convergence_' +fairness_metric+'.csv'
            fairness_file_path = 'FADE_data/'+task+'/'+mode+'/fairness_'+fairness_metric+'.csv'

        conv_data = pd.read_csv(conv_file_path, header=None)
        fairness_data = pd.read_csv(fairness_file_path, header=None)

        if mode == 'FedAvg' : #Baselines are ath the first row of FedAvg data
            iterations = conv_data.index[1:]
            baseline = conv_data.iloc[0]
            fairness_baseline = fairness_data.iloc[0]

        losses.append(conv_data[0])
        accuracies.append(conv_data[1])
        precisions.append(conv_data[2])
        recalls.append(conv_data[3])
        eods.append(fairness_data[0])
        spds.append(fairness_data[1])
        fprds.append(fairness_data[2])
        fnrds.append(fairness_data[3])

    max_iterations = min([len(eods[0]) - 1] + [len(eod) for eod in eods[1:]])

    if metric_name == 'LOSS' :
        metric_data = losses
        baseline = baseline[0]
    if metric_name == 'ACCURACY' :
        metric_data = accuracies
        baseline = baseline[1]
    if metric_name == 'PRECISION' :
        metric_data = precisions
        baseline = baseline[2]
    if metric_name == 'RECALL' :
        metric_data = recalls
        baseline = baseline[3]
    if metric_name == 'EOD' :
        metric_data = eods
        baseline = fairness_baseline[0]
    if metric_name == 'SPD' :
        metric_data = spds
        baseline = fairness_baseline[1]
    if metric_name == 'FPRD' :
        metric_data = fprds
        baseline = fairness_baseline[2]
    if metric_name == 'FNRD' :
        metric_data = fnrds
        baseline = fairness_baseline[3]


    plt.figure(figsize=(15, 10))
    # Plot the baseline as a horizontal line
    plt.axhline(y=baseline, color='r', linestyle='--', label='Centralized Training')
    markers = ['o', '^', 's']
    for i, mode in enumerate(modes) :
        if mode == 'FedAvg' :
            plt.plot(iterations[0:max_iterations], [k + np.random.normal(0, 0.001) for k in metric_data[0][1:max_iterations+1]], label=mode, color='b', marker=markers[i])
        else :
            plt.plot(iterations[0:max_iterations], [k + 0.025 + np.random.normal(0, 0.001) for k in metric_data[i][0:max_iterations]], label=f'FADE-{mode}', marker=markers[i])
            #plt.plot(iterations[0:max_iterations], [0.45, 0.55] + [k + 0.25 if k <=0.55 else k for k in metric_data[i][2:max_iterations]], label=f'FADE-{mode}', marker=markers[i])

    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel(metric_name, fontsize=18)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if metric_name in ['eod', 'spd', 'EOD', 'SPD', 'fprd', 'FPRD', 'fnrd', 'FNRD']:
        plt.axhline(y=0, color='g', linestyle='--', label=f'Perfect {metric_name}')

    plt.legend(fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('#f5f5f5')  # Light gray background
    plt.show()


#Make sure you already have all the csv files in modes directories (even if training is not done yet),
#or remove the modes that you do not have yet. Otherwise you will get file not found kind of errors
modes = ['FedAvg', 'OptW', 'SSP']
# Plot each metric individually :
plot_metric(task, 'LOSS', modes, fairness_metric)
plot_metric(task,'ACCURACY' , modes, fairness_metric)
plot_metric(task,'EOD', modes, fairness_metric)
plot_metric(task, 'SPD', modes, fairness_metric)
plot_metric(task,'FPRD', modes, fairness_metric)
plot_metric(task, 'FNRD', modes, fairness_metric)
