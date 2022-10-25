import os 
import warnings
import pandas as pd
import numpy as np
from sys import argv
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import *

### Useful functions
def models_pycaret(train_set, test_set, path, directory):
    """
    Create and get metrics for the best 5 Machine Learning models

    Parameters
    ----------
    train_set: pandas dataframe
        Values fron train the machine learning models
    test_set: pandas dataframe
        Values for test the machine leraning models
    path: dir
        Directory to save models

    Return
    ------
    metrics: pandas dataframe
        Table with 7 different metrics (Accuracy, AUC, Recall, Precision, F1, Kappa coeficient, MCC, Balanced acuraccy) for evaluate Machine Leraning behaviour 
    """
    ### Charge info to process
    best_models = setup(data = train_set, target = "activity", session_id = 125, log_experiment = False,
                    normalize = True, fold_shuffle=True, silent = True,
                    fix_imbalance=True, fix_imbalance_method= adasyn1)
    ### Create the best models, tune and finalize them
    models = compare_models(
        sort = "f1", include=["rf", "xgboost", "svm", "mlp"],
        n_select=7
    )
    info = pull()
    tuned_models = [tune_model(model, optimize= "f1") for model in models]
    fin_models = [finalize_model(model) for model in tuned_models]

    ### Save the models
    folder = f"{path}/models/{directory}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    names = info.index.tolist()
    for model, id in zip(fin_models, names):
        save_model(model, f"{folder}/{id}")   

    data_model = []
    bal_acc = []
    prec_recall_auc= []
    for model in fin_models:
        un_pre = predict_model(model, data= test_set)
        all_metrics = pull()
        data_model.append(all_metrics)
        bal_acc.append(round(balanced_accuracy_score(un_pre["activity"], un_pre["Label"]),4))
        prec_recall_auc.append(round(average_precision_score(un_pre["activity"], un_pre["Label"]),4))

    metrics = pd.concat(data_model).reset_index(drop = True)
    metrics["Balanced accuracy"] = bal_acc
    metrics["Prec_recall_auc"] = prec_recall_auc

    return metrics

### Avoid warnings
warnings.filterwarnings("ignore")

### Create a list from diferent folders with PLIFs files
dir_list = [dir[0] for dir in os.walk(argv[1])]
dir_list.pop(0)
target_lst = [val.split("/")[-1] for val in dir_list]

### Open oversampling method
adasyn1= ADASYN(sampling_strategy="minority")

### Fingerprint list
fp_list = ["PADIF", "PADIF2", "PROLIF", "ECIF"]

for row in zip(dir_list, target_lst):
    metrics = []
    for fp in fp_list:
        df = pd.read_csv(f"{row[0]}/{row[1]}_{fp}.csv", sep = ",", index_col=0)
        print(f"reading {row[1]}_{fp}")

        def act_change(val):
            if str(val) == "Active":
                act = 1
            else:
                act = 0
            return act
        
        df.columns = [*df.columns[:-1], "activity"]
        df["activity"] = df["activity"].apply(lambda x: act_change(x))

        y = df.pop("activity").to_frame()
        x = df

        X_train, X_test, y_train, y_test =train_test_split(
                x, y,stratify=y, test_size=0.15)
        data = pd.concat([X_train, y_train], axis=1)
        data = data.drop(data.columns[0], axis=1)
        data_unseen = pd.concat([X_test, y_test], axis=1)
        data_unseen = data_unseen.drop(data_unseen.columns[0], axis=1)

        metric = models_pycaret(data, data_unseen, row[0], fp)
        metric["FP"] = fp
        metrics.append(metric)
        
        print(f"ending {row[1]}_{fp}")

    table = pd.concat(metrics)
    table.to_csv(f"{row[0]}/{row[1]}_metrics_f.csv", sep=",", index= False)