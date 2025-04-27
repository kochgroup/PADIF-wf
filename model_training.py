"""
Python script for split and train machine learning models 
"""

import os
import pandas as pd
from tqdm import tqdm
from deepchem import data, splits
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import *

### Define splitness funtions

### Create a funtion for randoms splitness
def random_stratify_split(df):
    ### Select data and split
    y = df.pop("activity").to_frame()
    x = df

    X_train, X_test, y_train, y_test =train_test_split(
            x, y,stratify=y, test_size=0.1)
    data = pd.concat([X_train, y_train], axis=1)
    data_unseen = pd.concat([X_test, y_test], axis=1)
    return data, data_unseen

def molecular_stratify_split(df, molecular_method):
    x = df.loc[:, df.columns != 'activity'].values
    y = df[['activity']].astype('float')['activity'].values

    dataset = data.DiskDataset.from_numpy(
        X = x,
        y = y,
        w = x,
        ids = df.smiles.tolist()
    )
    if molecular_method == 'scaffold':
        scf_split = splits.ScaffoldSplitter()
        train, test = scf_split.train_test_split(dataset, frac_train=0.9)
    elif molecular_method == 'fingerprint':
        fp_split = splits.FingerprintSplitter()
        train, test = fp_split.train_test_split(dataset, frac_train=0.9)
    else:
        print('Incorrect molecular splitter')
    
    train_f = df[df.smiles.isin(train.ids.tolist())]
    test_f = df[df.smiles.isin(test.ids.tolist())]

    
    return train_f, test_f


import warnings
warnings.simplefilter('ignore', pd.errors.DtypeWarning)

def to_split(target, padif_folder, path_to_work, tp, method='random'):

    if method == 'random':
        splitter = random_stratify_split
    else:
        splitter = molecular_stratify_split

    
    print(f'''
        ********************************************************************************
        Startng to process with {target}, with {method} splitter
        ********************************************************************************
    \n''')
    ### split data
    set = pd.read_parquet(f'{padif_folder}/{target}_{tp}.parquet')
    if method == 'scaffold':
        actives = set[set['activity'] == 1]
        decoys = set[set['activity'] == 0]
        data_act, data_unseen_act = splitter(df=actives, molecular_method = 'scaffold')
        data_dec, data_unseen_dec = splitter(df=decoys, molecular_method = 'scaffold')
        data = pd.concat([data_act, data_dec])
        data_unseen = pd.concat([data_unseen_act, data_unseen_dec])
    elif method == 'fingerprint':
        actives = set[set['activity'] == 1]
        decoys = set[set['activity'] == 0]
        data_act, data_unseen_act = splitter(df=actives, molecular_method = 'fingerprint')
        data_dec, data_unseen_dec = splitter(df=decoys, molecular_method = 'fingerprint')
        data = pd.concat([data_act, data_dec])
        data_unseen = pd.concat([data_unseen_act, data_unseen_dec])
    else:
        data, data_unseen = splitter(df=set)

    print(
    f'{method} set\n'
    f'actives proportion in training set is: {round((data.activity.value_counts()[1] / len(data)), 3)}\n'
    f'actives proportion in test set is: {round((data_unseen.activity.value_counts()[1] / len(data_unseen)), 3)}'
    )
   
    ### save dataframes
    data.to_parquet(f'{path_to_work}/Train.parquet', sep=',', index=False)
    data_unseen.to_parquet(f'{path_to_work}/Test.parquet', sep=',', index=False)


adasyn1= ADASYN(sampling_strategy="minority")

## Funbction for train models
def models_pycaret(train_set, path):
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
        Table with 7 different metrics (Accuracy, AUC, Recall, Precision, F1, Kappa coeficient, MCC, Balanced acuraccy) 
        for evaluate ML models 
    """
    ### Charge info to process
    train = pd.read_parquet(train_set)
    train = train.drop(["id", "score", "smiles"], axis=1)
    train = train.dropna()
    train = train.astype('float64')
    ### mesure active balance in dataset
    if (train.activity.value_counts()[1] / len(train)) <= 0.35:
        best_models = setup(data = train, target = "activity", session_id = 125, log_experiment = False,
                        normalize = True, fold_shuffle=True,
                        fix_imbalance=True, fix_imbalance_method= adasyn1)
    else:
        best_models = setup(data = train, target = "activity", session_id = 125, log_experiment = False,
                        normalize = True, fold_shuffle=True)
    ### Create the best models, tune and finalize them
    models = compare_models(
        sort = "f1", include=["rf", "xgboost", "svm", "mlp"],
        n_select=4
    )
    info = pull()
    tuned_models = [tune_model(model, optimize= "f1") for model in models]
    fin_models = [finalize_model(model) for model in tuned_models]

    ### Save the models
    names = info.index.tolist()
    for model, id in zip(fin_models, names):
        save_model(model, f"{path}/{id}")   

def padif_train(target, splitter_types, path_to_work):
    
    model_dir = os.path.join(path_to_work, 'models')
    os.makedirs(model_dir, exist_ok=True)

    for type in splitter_types:
        folder = f'{path_to_work}/data_to_model/{type}'
        folder2 = os.path.join(model_dir, type)
        os.makedirs(folder2, exist_ok=True)
        print(
        f'''    
        ********************************************************************************
        
        Starting to work with {target}, splitter {type}
        
        ********************************************************************************
        '''
        )  
        models_pycaret(f'{folder}/Train.parquet', folder2)

def split_datasets(splitters, path_target, target_name, tp):

    data_to_model = os.path.join(path_target, 'data_to_model')
    os.makedirs(data_to_model, exist_ok=True)
    ### Split with all types of splitters
    for splitter in splitters:
        spl_folder = os.path.join(f'{data_to_model}', splitter)
        os.makedirs(spl_folder, exist_ok=True)
        path = f'{data_to_model}/{splitter}'
        to_split(target=target_name, padif_folder=path_target, path_to_work=path, tp=tp, method=splitter)


### Split and train
df = pd.read_parquet("files/targets_information.parquet")

types = ['true', 'dcm', 'znc', 'div']
   
splitters = ['random', 'scaffold', 'fingerprint']

for tp in types:
    for name in df.name:
        folder = f'files/{name}/{tp}'
        split_datasets(splitters, folder, name, tp)
        padif_train(name, splitters, folder)    
