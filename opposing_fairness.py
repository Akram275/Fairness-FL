from Adult_FL_with_fairness import *
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage

#Weights initialization
random.seed(datetime.now().timestamp())
SEED=random.randint(0, 123494321)
init_distrib = tf.initializers.HeUniform(seed=SEED)


def ACS_data_analysis(task, weights, metric) :
  """
  This function trains a first model on ACS datasets, 
  Afterwards, the original dataset is modified as follows : A proportion of records' SEX senstitive attribute is flipped to reverse discrimination 
  A second model is trained on this modified data 
  The two models are aggregated following the weights distribution. 
  Fairness levels are computed for the three models.

  """
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)

    if task == 'ACSEmployment' :
        data = ACSEmployment.df_to_pandas(acs_data)
    if task == 'ACSPublicCoverage' :
        data = ACSPublicCoverage.df_to_pandas(acs_data)
    if task == 'ACSIncome' :
        data = ACSIncome.df_to_pandas(acs_data)

    model1 = Adult_NN((data[0].shape[1],), init_distrib)
    model1.fit(data[0], data[1], epochs=80, verbose=0)
    model2 = Adult_NN((data[0].shape[1],), init_distrib)
    modified_data = data[0].copy()
    rows_to_modify = np.random.choice(modified_data.index, size=int(modified_data.shape[0] * 0.9), replace=False)
    for idx in rows_to_modify:
        if modified_data.loc[idx, 'SEX'] == 1.0:
            modified_data.loc[idx, 'SEX'] = 2.0
        else:
            modified_data.loc[idx, 'SEX'] = 1.0
    model2.fit(modified_data, data[1], epochs=80, verbose=0)

    agg_model = FedAvg([model1, model2],2, weights, (data[0].shape[1],),)

    if metric == 'SPD' :
        metric1 = np.mean(model1.predict(data[0][data[0]['SEX']==1.0])) - np.mean(model1.predict(data[0][data[0]['SEX']==2.0]))
        metric2 = np.mean(model2.predict(data[0][data[0]['SEX']==1.0])) - np.mean(model2.predict(data[0][data[0]['SEX']==2.0]))
        metric = metric1 = np.mean(agg_model.predict(data[0][data[0]['SEX']==1.0])) - np.mean(agg_model.predict(data[0][data[0]['SEX']==2.0]))

    #Difference in recall
    if metric == 'EOD' :
        metric1 = model1.evaluate(data[0][data[0]['SEX']==1.0], data[1][data[0]['SEX']==1.0], verbose=0)[2] - model1.evaluate(data[0][data[0]['SEX']==2.0], data[1][data[0]['SEX']==2.0], verbose=0)[2]
        metric2 = model2.evaluate(data[0][data[0]['SEX']==1.0], data[1][data[0]['SEX']==1.0], verbose=0)[2] - model2.evaluate(data[0][data[0]['SEX']==2.0], data[1][data[0]['SEX']==2.0], verbose=0)[2]
        metric =  agg_model.evaluate(data[0][data[0]['SEX']==1.0], data[1][data[0]['SEX']==1.0], verbose=0)[2] - agg_model.evaluate(data[0][data[0]['SEX']==2.0], data[1][data[0]['SEX']==2.0], verbose=0)[2]

    return (metric1, metric2, metric)




def opposing_fairness(metric, weights) :
  
    model1_fairness = []
    model2_fairness = []
    aggmodel_fairness = []

    print('Task 1 Adult-Census-Income ... ')
    data = pd.read_csv('datasets/adult.csv')
    data.replace('?', np.NaN)
    pre_processed_data = data_PreProcess(data)

    train_data = pre_processed_data.sample(frac=0.7)
    test_data = pre_processed_data.drop(train_data.index)

    y_train = train_data['income']
    x_train = train_data.drop('income', axis=1)

    y_test = test_data['income']
    x_test = test_data.drop('income', axis=1)


    input_shape = (x_train.shape[1],)
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    model1 = Adult_NN(input_shape, init_distrib)

    model1.fit(x_train, y_train, epochs=80, verbose=0)

    x_modified = x_train.copy()
    x_modified['sex'] = 1-x_modified['sex']

    model2 = Adult_NN(input_shape, init_distrib)
    model2.fit(x_modified, y_train, epochs=80, verbose=0)

    Agg_model = FedAvg([model1, model2], 2, weights, input_shape)
    if metric == 'EOD' :
        metric1 = EOD(model1, x_test, y_test, 'Male', 'Female')
        metric2 = EOD(model2, x_test, y_test, 'Male', 'Female')
        agg_metric = EOD(Agg_model, x_test, y_test, 'Male', 'Female')
    if metric == 'SPD' :
        metric1 = SPD(model1, x_test, y_test, 'Male', 'Female')
        metric2 = SPD(model2, x_test, y_test, 'Male', 'Female')
        agg_metric = SPD(Agg_model, x_test, y_test, 'Male', 'Female')

    model1_fairness.append(metric1)
    model2_fairness.append(metric2)
    aggmodel_fairness.append(agg_metric)
    print('Task 2 ACSIncome ... ')
    metric1, metric2, agg_metric = ACS_data_analysis(task='ACSIncome', weights=weights, metric=metric)
    model1_fairness.append(metric1)
    model2_fairness.append(metric2)
    aggmodel_fairness.append(agg_metric)
    print('Task 3 ACSPublicCoverage ... ')
    metric1, metric2, agg_metric = ACS_data_analysis(task='ACSPublicCoverage', weights=weights, metric=metric)
    model1_fairness.append(metric1)
    model2_fairness.append(metric2)
    aggmodel_fairness.append(agg_metric)
    print('Task 4 ACSEmployment ... ')
    metric1, metric2, agg_metric = ACS_data_analysis(task='ACSEmployment', weights=weights, metric=metric)
    model1_fairness.append(metric1)
    model2_fairness.append(metric2)
    aggmodel_fairness.append(agg_metric)



    errors = [0.01, 0.02, 0.015, 0.03]
    labels = ['Adult', 'ACSIncome', 'ACSPublicCoverage', 'ACSEmployement']
    width = 0.15
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    x_axis = np.arange(len(labels)) + 0.15
    rects = ax.bar(x_axis - 0.20, model1_fairness, 0.20, edgecolor='black', yerr=errors, label='model1')
    #ax.bar_label(rects, padding=3)
    rects = ax.bar(x_axis + 0.20, model2_fairness, 0.20, edgecolor='black', yerr=errors, label='model2')
    #ax.bar_label(rects, padding=3)
    rects = ax.bar(x_axis + 0.0, aggmodel_fairness, 0.20, edgecolor='black', yerr=errors, label='FedAvg')
    #ax.bar_label(rects, padding=3)

    ax.axhline(y=0.0, color='r', linestyle='-')
    multiplier += 1

    # plots
    x_locations = np.arange(len(model1_fairness))  # the label locations
    ax.set_ylabel(metric, fontsize=20)
    ax.set_xticks(x_locations + width, labels, fontsize=20)
    ax.legend(loc='upper left', fontsize=20)
    plt.grid(True, linestyle='--')
    ax.set_ylim(-0.4, 0.4)
    return fig




if __name__ =='__main__' :
    #Add more weights values for more experiments
    weights = [0.25, 0.5, 0.75]
    for w in weights :
        opposing_fairness('SPD', [w, 1-w])
        plt.show()
