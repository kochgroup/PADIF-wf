#%%
import os
from glob import glob
import pandas as pd
import random
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

def padif_generator(file, cavity_atoms_file, docking_sln=1):
    """
    Extract chemplp for multiple docking solutions files

    Parameters
    ----------
    file: str
        Names of docking solution files for processing
    docking_sln: int
        number of docking solution

    Return
    ------
    df: pandas dataframe
        Dataframe with all interactions
    """
    ### Open solution file and extract protein contributions and score
    with open(file, 'r') as nfile:
        f = nfile.read()
    protein_plp = f.split('> <Gold.PLP.Protein.Score.Contributions>')[docking_sln].split('$$$$')[0].strip().split('\n')
    plp_dataframe = pd.DataFrame([line.split() for line in protein_plp])
    plp_dataframe.columns = plp_dataframe.iloc[0].tolist()
        
    ### Delete and change type of some rows and columns
    plp_dataframe = plp_dataframe[plp_dataframe["AtomID"] != "AtomID"].astype('float64')
    plp_dataframe = plp_dataframe.astype({'AtomID':int})
    plp_dataframe = plp_dataframe.drop('PLP.total', axis=1)

    ### Select only atoms in cavity atoms file
    with open(cavity_atoms_file, 'r') as nfile:
        cav_f = nfile.read()
    atoms_list = [int(l) for subl in [val.split() for val in  cav_f.split('\n')] for l in subl]
    plp_dataframe = plp_dataframe[plp_dataframe["AtomID"].isin(atoms_list)]

    ### change the value for the fisrt 3 columns
    plp_dataframe["ChemScore_PLP.Hbond"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]
    plp_dataframe["ChemScore_PLP.CHO"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]
    plp_dataframe["ChemScore_PLP.Metal"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]

    ### Create a interaction dataframe
    atomid_values = plp_dataframe.AtomID.tolist() 
    new_column_names = [f"{col}_{atomid}" for atomid in atomid_values for col in plp_dataframe.columns if col != 'AtomID']
    interaction = plp_dataframe.drop(columns=['AtomID'])
    interaction_list = [val for sublist in interaction.values.tolist() for val in sublist]
    interaction_df = pd.DataFrame([interaction_list], columns=new_column_names)

    ### Add score and id for each 
    interaction_df['score'] = round(float(f.split('> <Gold.PLP.Fitness>')[1].split('> <Gold.PLP.PLP>')[0].strip()),3)

    if docking_sln != 1:
        interaction_df['id'] = file.split('/')[-1].replace("_sln.sdf", "") + "_" + str(docking_sln)
    else:
        interaction_df['id'] = file.split('/')[-1].replace("_sln.sdf", "").replace('-1', '')

    return interaction_df

def organize_padif(out, activity_value):
    padif_df = pd.concat((df.T for df in out), axis=1).T
    padif_df = padif_df.fillna(0.0)
    to_move = ['score', 'id']
    order_1 =[col for col in padif_df.columns if col not in to_move] + to_move
    padif_df = padif_df[order_1]
    padif_df_copy = padif_df.copy()
    new_columns_df = pd.DataFrame({'activity': activity_value}, index=padif_df_copy.index)
    padif_df = pd.concat([padif_df_copy, new_columns_df], axis=1)
    padif_df = padif_df.reset_index(drop=True)

    return padif_df

def padif_to_dataframe(folder, cavity_file, actives=False):
    """
    Convert GOLD docking solutions in PADIF

    Parameters:
    -----------
    folder: str
        path with GOLD docking solution to process
    actives: bool
        define if your molecules are actives or decoys 
    diverse_sln: bool
        define if you need to use other docking conformations 
        (from diverse solutions or not) to create PADIF
    n_sln: int
        number or docking conformations to have into account,
        specialy useful for diverse_sln process
    n_mols: int
        number of molecules that you need to include in your 
        PADIF
    random_seed: int
        for decoy random selection, this number guarranty the
        reproducibility 
    """

    files = glob(f'{folder}/*_sln.sdf')
    ### extract padif from actives
    params = zip(files, [cavity_file]*len(files))
    out = Parallel(n_jobs=-1)(delayed(padif_generator)(file, cavity) for file, cavity in params)
    if type == 'actives':
        ### organize and add acitivity column
        padif_df = organize_padif(out, 1)
    
    elif type == 'decoys':
        ### organize and add acitivity column
        padif_df = organize_padif(out, 0)
    
    else:
        ### organize and add acitivity column
        padif_df = organize_padif(out, 0)
        padif_df = padif_df.drop(columns=['activity'])
    
    return padif_df
