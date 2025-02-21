#%%
import os
import gc
import ast
import glob
import random
import shutil
import warnings
import tempfile
import subprocess
import numpy as np
import pandas as pd
import time
import deepchem
from sys import argv
from joblib import Parallel, delayed
from docking_tools import chembl_mols, prep_ligand_from_smiles, gold_config, dock_mol, gen_scaffold, protein_prepare
from fp_extract import padif_to_dataframe
from split_datasets import to_split
from ml_training import padif_train
from metrics import predictions_models, figures_from_metrics
from residue_atom_info import atom_interaction_info, atoms_by_activity, residue_interactions, write_chimera_commands
#%%
"""
Starting calculations
"""
### Initial parameters
njobs = int(20)
# njobs = os.cpu_count()
warnings.filterwarnings("ignore")

### time
start_cpu = time.process_time()
start_clock = time.time()

def chembl_download(chembl_code):

    ### Download the molecules and create datframes from ChEMBL
    dfActivities, targetName = chembl_mols('CHEMBL4760')

    ### Delete bad strings in target name
    unaceptable_strings = ["/", "(", ")", ",", ";", ".", " "]
    for string in unaceptable_strings:
        targetName = targetName.replace(string, "_")

    ligands = dfActivities[["id", "new_smiles"]]

    return ligands, targetName

def protein_process(protein, ligand_id, path_target, target_name):    

    ### Open and prepare protein 
    prot1 = protein

    ### preprare protein and ligand
    protein_prepare(
        protein=prot1,
        path=path_target,
        target_name=target_name,
        save_ligand=True,
        ligand_id=ligand_id,
        multiple_chain=False,
        chain_to_work='A',
        remove_metals=False
    )

def active_docking(ligands, path_target, target_name):

    ### Create temporary folder to work
    sysTemp = tempfile.gettempdir()
    myTemp = os.path.join(sysTemp,'mytemp')
    #You must make sure myTemp exists
    if not os.path.exists(myTemp):
        os.makedirs(myTemp)

    ### Make a temp folder and prepare ligands
    act_dir = tempfile.mkdtemp(suffix=None,prefix='act_',dir=myTemp)
    ligands['act_dir'] = [act_dir]*len(ligands)
    tasks = ligands.apply(lambda row: (row['new_smiles'], row['id'], row['act_dir']), axis=1)
    results = Parallel(n_jobs=-1)(delayed(prep_ligand_from_smiles)(*task) for task in tasks)

    ### create the gold config for docking
    gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                ref_ligand=f"{path_target}/Ligand_{target_name}.mol2",
                gold_name=f'{target_name}_gold',
                path=path_target)

    ### Dock the Active compounds
    to_dock = ligands[['id', 'act_dir']]
    to_dock['gold_file'], to_dock['num_sln'] = [f'{path_target}/{target_name}_gold.conf']*len(to_dock), [10]*len(to_dock)
    tasks_dock = to_dock.apply(lambda row: (row['gold_file'], row['id'], row['act_dir'], row['num_sln']), axis=1)
    results_dock = Parallel(n_jobs=-1)(delayed(dock_mol)(*task) for task in tasks_dock)

    ### Create active and inactive directories and save docking files
    act_f = os.path.join(path_target, "Actives")
    os.makedirs(act_f, exist_ok=True)

    for filename in glob.glob(act_dir + f"/*_sln.sdf"):
        shutil.copy(filename, act_f)

    ### Copy plp_protein to principal folder
    shutil.copy(f"{act_dir}/plp_protein.mol2", path_target)  

    return myTemp, act_f  

def decoys_docking(ligands, temp, path_target, target_name):

    ### Calculate scaffolds for active compounds
    scaffolds = Parallel(n_jobs=-1)(delayed(gen_scaffold)(smi) for smi in ligands.new_smiles.values)
    ligands['scf'] = scaffolds
    ligands = ligands[(ligands['scf'] != 'Error-Smiles') & (ligands['scf'] != 'Something else')]    

    ### Select different compounds as decoys using scaffolds
    decoys_df = pd.read_csv('./files/DCM_prepared.csv', sep=',')  
    bad_id_dec = pd.merge(decoys_df, ligands, on="scf")["id_x"].unique().tolist()
    decoys_df_f = decoys_df[~decoys_df.id.isin(bad_id_dec)]
    decoys = decoys_df.iloc[random.sample(range(len(decoys_df_f)), len(ligands)*4)] 

    ### make a temp folder and prepare decoys
    dec_dir = tempfile.mkdtemp(suffix=None,prefix='ina_',dir=temp)
    decoys['ina_dir'] = [dec_dir]*len(decoys)
    tasks_decoys = decoys.apply(lambda row: (row['smiles'], row['id'], row['ina_dir']), axis=1)
    results = Parallel(n_jobs=-1)(delayed(prep_ligand_from_smiles)(*task) for task in tasks_decoys) 

    ### Dock the Active compounds
    to_dock_decoys = decoys[['id', 'ina_dir']]
    to_dock_decoys['gold_file'], to_dock_decoys['num_sln'] = [f'{path_target}/{target_name}_gold.conf']*len(to_dock_decoys), [10]*len(to_dock_decoys)
    tasks_dock_decoys = to_dock_decoys.apply(lambda row: (row['gold_file'], row['id'], row['ina_dir'], row['num_sln']), axis=1)
    results_dock_decoys = Parallel(n_jobs=-1)(delayed(dock_mol)(*task) for task in tasks_dock_decoys)   

    ### Create decoys directory, save docking files and copy protein
    dec_f = os.path.join(path_target, "Decoys")
    os.makedirs(dec_f, exist_ok=True)   

    for filename in glob.glob(dec_dir + f"/*_sln.sdf"):
        shutil.copy(filename, dec_f)    

    shutil.copy(f"{dec_dir}/plp_protein.mol2", path_target)

    return dec_f

def padif_extraction(ligands, active_folder, decoy_folder, path_target, target_name):

    ### Extract PADIF for actives compounds and add the SMILES for each molecule
    active_padif = padif_to_dataframe(active_folder, cavity_file=f'{path_target}/cavity.atoms', actives=True)
    ligands_info = ligands[['id','new_smiles']]
    active_padif = active_padif.merge(ligands_info, on='id')
    active_padif = active_padif.rename(columns={'new_smiles':'smiles'})
    ### Extract PADIF for decoys compounds and add the SMILES for each molecule
    decoys_padif = padif_to_dataframe(decoy_folder, cavity_file=f'{path_target}/cavity.atoms')
    decoys_info = decoys[['id','smiles']]
    decoys_padif = decoys_padif.merge(decoys_info, on='id')
    ### concat the list and save PADIF file
    final_padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
    final_padif = final_padif.fillna(0.0)
    ### Sort columns and put in the end of dataframe for better visualization
    to_move = ['score', 'id', 'smiles','activity']
    order_1 =[col for col in final_padif.columns if col not in to_move] + to_move
    final_padif = final_padif[order_1]
    final_padif.to_csv(f'{path_target}/{target_name}_PADIF.csv', sep=',', index=False)

def split_datasets(splitters, path_target, target_name):

    data_to_model = os.path.join(path_target, 'data_to_model')
    os.makedirs(data_to_model, exist_ok=True)
    ### Split with all types of splitters
    for splitter in splitters:
        spl_folder = os.path.join(data_to_model, splitter)
        os.makedirs(spl_folder, exist_ok=True)
        path = f'{data_to_model}/{splitter}'
        to_split(target=target_name, padif_folder=path_target, path_to_work=path, method=splitter)
   

def main(chembl_code, protein_file, ligand_id):

    ligands, target_name = chembl_donwload(chembl_code)
    
    print(
        f'''    
        ********************************************************************************
        
        Starting to work with {target_name} docking process
        
        ********************************************************************************
        '''
    )

    ### Protein preparation
    parenr_dir = os.getcwd()
    path_target = os.path.join(f"{parenr_dir}/files", target_name)
    os.makedirs(path_target, exist_ok=True)

    protein_process(protein_file, ligand_id, path_target, target_name)

    temp, active_folder = actives_docking(ligands, path_target, target_name)
    
    decoy_folder = decoys_docking(ligands, temp, path_target, target_name)

    padif_extraction(ligands, active_folder, decoy_folder, path_taget, target_name)

    splitters = ['random', 'scaffold', 'fingerprint']

    split_datasets(splitters, path_target, target_name)

    ### Train models
    padif_train(target_name, splitters, path_target)

    ### Save figures and statistics
    metrics_df, predictions_df = predictions_models(splitters, path_target)
    figures_from_metrics(predictions_df, metrics_df, path_target)

    metrics_df.to_csv(f'{path_target}/metrics_of_models.csv', sep=',', index=False)
    predictions_df.to_csv(f'{path_target}/predictions_of_models.csv', sep=',', index=False)

    ### Remove Bad files
    gc.collect()
    shutil.rmtree(temp)

if __name__ == '__main__':
    main(argv[1], argv[2], argv[3])



# #%%
# ### Save all atoms info and display in chimera

# ### Extract information about protein and atoms by activity
# protein_df = atom_interaction_info(path_target)
# act, dec, all = atoms_by_activity(final_padif, protein_df)
# ### Create folder to save residue and atom information
# atoms_path = os.path.join(path_target, 'atoms_data')
# os.makedirs(atoms_path, exist_ok=True)
# ### Save information about residue
# res_info = residue_interactions(final_padif, atoms_path)
# res_info.to_csv(f'{atoms_path}/residue_information.csv', sep=',', index=False)
# ### Create command file and start chimera
# # Path to the command file you want to create
# command_file_path = f"{atoms_path}/setup_chimera.cmd"
# # Create the Chimera command file
# write_chimera_commands(
#      f'{path_target}/{targetName}_prep.mol2',
#      list(all),
#      list(act),
#      list(dec),
#      f'{atoms_path}',
#      command_file_path
# )
# ### Start process
# ch_path = '/appl/UCSF/Chimera64-1.17.3/bin/chimera'
# subprocess.run([ch_path, command_file_path])
