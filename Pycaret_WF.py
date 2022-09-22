import os 
import pandas as pd
from sys import argv
from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import *

### Useful functions
def models_pycaret(train_set, test_set):
    """
    Create and get metrics for the best 5 Machine Learning models

    Parameters
    ----------
    train_set: pandas dataframe
        Values fron train the machine learning models
    test_set: pandas dataframe
        Values for test the machine leraning models

    Return
    ------
    metrics: pandas dataframe
        Table with 7 different metrics (Accuracy, AUC, Recall, Precision, F1, Kappa coeficient, MCC, Balanced acuraccy) for evaluate Machine Leraning behaviour 
    """
    ### Charge info to process
    best_models = setup(data = train_set, target = "activity", session_id = 125, log_experiment = False,
                    normalize = True, transformation = True, 
                    ignore_low_variance = True, fold_shuffle=True, silent = True,
                    fix_imbalance=True, fix_imbalance_method= adasyn1, use_gpu = True,
                    imputation_type='iterative')
    ### Create the best models, tune and finalize them
    top5 = compare_models(n_select=5, sort= "f1", exclude=["dummy"])
    tuned_top5 = [tune_model(model, optimize= "f1") for model in top5]
    fin_top5 = [finalize_model(model) for model in tuned_top5]    

    data_model = []
    bal_acc = []
    for model in fin_top5:
        un_pre = predict_model(model, data= test_set)
        all_metrics = pull()
        data_model.append(all_metrics)
        bal_acc.append(round(balanced_accuracy_score(un_pre["activity"], un_pre["Label"]),4))

    metrics = pd.concat(data_model).reset_index(drop = True)
    metrics["Balanced accuracy"] = bal_acc

    return metrics

### Create a list from diferent folders with PLIFs files
dir_list = [dir[0] for dir in os.walk(argv[1])]
dir_list.pop(0)
target_lst = [val.split("/")[-1] for val in dir_list]

### Open oversampling method
adasyn1= ADASYN(sampling_strategy="minority")

### Fingerprint list
fp_list = ["PADIF", "PADIF2", "PROLIF", "ECIF"]

for row in zip(dir_list, target_lst):
    for fp in fp_list:
        df = pd.read_csv(f"{row[0]}/{row[1]}_{fp}.csv", sep = ",", index_col=0)

        def act_change(val):
            if str(val) == "Active":
                act = 1
            else:
                act = 0
            return act
        
        df.columns = [*df.columns[:-1], "activity"]
        df["activity"] = df["activity"].apply(lambda x: act_change(x))

        data = df.sample(frac=0.95, random_state=786)
        data_unseen = df.drop(data.index)

        data.reset_index(inplace=True, drop=True)
        data_unseen.reset_index(inplace=True, drop=True)

        metrics = []
        for fp in fp_list:
            metric = models_pycaret(data, data_unseen)
            metric["FP"] = fp
            metrics.append(metric)

        table = pd.concat(metrics)
        table.to_csv(f"{row[0]}/{row[1]}_metrics.csv", sep=",", index= False)