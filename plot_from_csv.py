import pandas as pd
import matplotlib.pyplot as plt
import sys
# Read the CSV file without headers
task = sys.argv[1]




def plot_metric(task, metric_name, modes):
    accuracies = []
    losses = []
    precisions = []
    recalls = []
    eods = []
    spds = []

    for mode in modes :
        conv_file_path = 'CIKM_data/'+task+'/'+mode+'/convergence.csv'
        fairness_file_path = 'CIKM_data/'+task+'/'+mode+'/fairness.csv'

        conv_data = pd.read_csv(conv_file_path, header=None)
        fairness_data = pd.read_csv(fairness_file_path, header=None)

        if mode == 'FedAvg' :
            iterations = conv_data.index[1:]
            baseline = conv_data.iloc[0]
            fairness_baseline = fairness_data.iloc[0]
        losses.append(conv_data[0])
        accuracies.append(conv_data[1])
        precisions.append(conv_data[2])
        recalls.append(conv_data[3])
        eods.append(fairness_data[0])
        spds.append(fairness_data[1])

    max_iterations = min([len(eod) for eod in eods])

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
    if metric_name == 'SPD' :
        metric_data = spds
        baseline = fairness_baseline[1]
    if metric_name == 'EOD' :
        metric_data = eods
        baseline = fairness_baseline[0]

    plt.figure(figsize=(8, 6))
    # Plot the baseline as a horizontal line
    plt.axhline(y=baseline, color='r', linestyle='--', label='Centralized Training')
    markers = ['o', '^', 's']
    for i, mode in enumerate(modes) :
        if mode == 'FedAvg' :
            plt.plot(iterations[0:max_iterations], metric_data[0][1:max_iterations+1], label=mode, color='b', marker=markers[i])
        else :
            print(len(iterations))
            print(len(metric_data[i]))
            plt.plot(iterations[0:max_iterations], [k for k in metric_data[i][0:max_iterations]], label=f'FADE-{mode}', marker=markers[i])

    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel(metric_name, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if metric_name in ['eod', 'spd', 'EOD', 'SPD']:
        plt.axhline(y=0, color='g', linestyle='--', label=f'Perfect {metric_name}')

    plt.legend(fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('#f5f5f5')  # Light gray background
    plt.show()


#Makes sure you already have all the csv files in modes directories (even if training is not done yet),
#or remove the modes that you do not have yet. Otherwise you will get file not found kind of errors
modes = ['FedAvg', 'WO', 'SSP']
# Plot each metric individually :
plot_metric(task, 'LOSS', modes)
plot_metric(task,'ACCURACY' , modes)
plot_metric(task,'EOD', modes)
plot_metric(task, 'SPD', modes)
