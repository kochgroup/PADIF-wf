"""
Python script for extract PADIF from docking solutions
"""

import os
from glob import glob
import pandas as pd
import random
from tqdm import tqdm
import multiprocessing

def padif_generator(file, docking_sln=1):
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
    protein_plp = f.split('> <Gold.PLP.Protein.Score.Contributions>').split('$$$$')[0].strip().split('\n')
    plp_dataframe = pd.DataFrame([line.split() for line in protein_plp])
    plp_dataframe.columns = plp_dataframe.iloc[0].tolist()
        
    ### Delete and change type of some rows and columns
    plp_dataframe = plp_dataframe[plp_dataframe["AtomID"] != "AtomID"].astype('float64')
    plp_dataframe = plp_dataframe.astype({'AtomID':int})
    plp_dataframe = plp_dataframe.drop('PLP.total', axis=1)

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

def padif_to_dataframe(folder, actives=False, diverse_sln=False, n_sln=20, n_mols=4, random_seed=9876):
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

    if actives == True:
        ### extract padif from actives
        if __name__ == '__main__':
            pool = multiprocessing.Pool()
            results = [pool.apply_async(padif_generator, (file,1)) for file in  glob(f'{folder}/*_sln.sdf')]
            pool.close()
            out = [result.get() for result in results]

        ### organize and add acitivity column
        padif_df = organize_padif(out, 1)

    elif diverse_sln == True:
        ### extract inactives from other solutions
        if __name__ == '__main__':
            random.seed(random_seed)
            out_div = []
            for num in random.sample(range(2,n_sln), n_mols):
                pool = multiprocessing.Pool()
                results = [pool.apply_async(padif_generator, (file, num)) for file in  glob(f'{folder}/*_sln.sdf')]
                pool.close()
                out_div.append([result.get() for result in results])
            out_total = [val for sublist in out_div for val in sublist]

        ### organize and add acitivity column
        padif_df = organize_padif(out_total, 0)

    else:
        ### Select from decoys folder a number to process     
        try:
            random.seed(random_seed)
            if __name__ == '__main__':
                pool = multiprocessing.Pool()
                results = [pool.apply_async(padif_generator, (file,)) for file in  random.sample(glob(f'{folder}/*_sln.sdf'), n_mols)]
                pool.close()
                out = [result.get() for result in results]

        except:
            if __name__ == '__main__':
                pool = multiprocessing.Pool()
                results = [pool.apply_async(padif_generator, (file,)) for file in  glob(f'{folder}/*_sln.sdf')]
                pool.close()
                out = [result.get() for result in results]

        ### organize and add acitivity column
        padif_df = organize_padif(out, 0)
    
    return padif_df


### Open target list
df = pd.read_parquet("files/target_information.parquet")

types_mols = [
    'div',
    'dcm',
    'znc', 
    'true'
]

for target in tqdm(df.iloc):
    ### select variables to work
    name = target['name']
    set = target['set']
    path = 'files/docking_sln'
    print(
        f'''
        ********************************************************************************
        Startng to process with {target['target']}
        ********************************************************************************

        '''
    )
    ### Create PADIF for actives 
    active_padif = padif_to_dataframe(f'files/docking_sln/{name}/actives_solution.sdf', 
                                      actives=True)
    ### Open a reference file with smiles and merge by id 
    active_reference = pd.read_parquet(f'files/chembl_data/{set}_{name}.parquet')
    active_reference = active_reference[['canonical_smiles', 'molecule_chembl_id']]
    active_reference.rename(columns={
        'canonical_smiles':'smiles', 
        'molecule_chembl_id':'id'
        }, inplace=True)
    active_padif = active_padif.merge(active_reference, on='id')

    for type in types_mols:
        print(
        f'''
        
        Creating PADIF for {target['target']} with {type} decoys
        
        '''
        )
        if type == 'true':
            ### Create PADIF for true inactives
            decoys_padif = padif_to_dataframe(f'files/docking_sln/{name}/inactives_solution.sdf',
                                              actives=False,
                                              n_mols=len(active_padif)*4)
            decoys_padif['id'] = decoys_padif['id'].astype('int')
            ### Open a reference file with smiles and merge by id 
            decoys_reference = pd.read_parquet(f'files/inactives/{name}_ina.parquet')
            decoys_padif = decoys_padif.merge(decoys_reference, on='id')

            padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
            padif = padif.fillna(0.0)
            ### Sort columns and put in the end of dataframe for better visualization
            to_move = ['score', 'id', 'smiles','activity']
            order_1 =[col for col in padif.columns if col not in to_move] + to_move
            padif = padif[order_1]
            padif.to_parquet(f'files/padif/{name}_{type}.parquet', sep=',', index=False)

        elif type == 'div':
            ### Create PADIF for Diverse solutions, for this It's necessary to open Actives solutions
            decoys_padif = padif_to_dataframe(f'files/docking_sln/{name}/actives_solution.sdf',
                                              actives=False,
                                              diverse_sln=True)
            decoys_padif['id2'] = [val.split('-')[0] for val in decoys_padif.id.tolist()]
            ### Open a reference file with smiles and merge by id 
            active_reference.rename(columns={'id':'id2'}, inplace=True)
            decoys_padif = decoys_padif.merge(active_reference, on='id2')
            decoys_padif = decoys_padif.drop(columns=['id2'])
            padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
            padif = padif.fillna(0.0)
            ### Sort columns and put in the end of dataframe for better visualization
            to_move = ['score', 'id', 'smiles','activity']
            order_1 =[col for col in padif.columns if col not in to_move] + to_move
            padif = padif[order_1]
            padif.to_parquet(f'files/padif/{name}_{type}.parquet', sep=',', index=False)
        
        elif type == 'dcm':
            ### Create a PADIF for Dark Chemical Matter Solutions 
            decoys_padif = padif_to_dataframe(f'files/docking_sln/{name}/decoys_{type}_solution.sdf',
                                              actives=False,
                                              n_mols=len(active_padif)*4)
            ### Open a reference file with smiles and merge by id 
            decoys_reference = pd.read_parquet('files/DCM_prepared.parquet',
                                           index_col=0)
            decoys_reference = decoys_reference[['smiles', 'id']]
            decoys_padif = decoys_padif.merge(decoys_reference, on='id')
            padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
            padif = padif.fillna(0.0)
            ### Sort columns and put in the end of dataframe for better visualization
            to_move = ['score', 'id', 'smiles','activity']
            order_1 =[col for col in padif.columns if col not in to_move] + to_move
            padif = padif[order_1]
            padif.to_parquet(f'files/padif/{name}_{type}.parquet', sep=',', index=False)

        else:
            ### Create PADIF for ZINC15 solutions
            decoys_padif = padif_to_dataframe(f'files/docking_sln/{name}/decoys_{type}_solution.sdf',
                                              actives=False,
                                              n_mols=len(active_padif)*4)
            ### Open a reference file with smiles and merge by id 
            decoys_reference = pd.read_parquet('files/znc_scf.parquet')
            decoys_reference = decoys_reference[['smiles', 'id']]
            decoys_padif = decoys_padif.merge(decoys_reference, on='id')
            padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
            padif = padif.fillna(0.0)
            ### Sort columns and put in the end of dataframe for better visualization
            to_move = ['score', 'id', 'smiles','activity']
            order_1 =[col for col in padif.columns if col not in to_move] + to_move
            padif = padif[order_1]
            padif.to_parquet(f'files/padif/{name}_{type}.parquet', sep=',', index=False)

