"""
Python file for extract chembl compounds, prepare and dock agiants specific targets
"""

import os
import gc
import glob
import random
import shutil
import warnings
import tempfile
import numpy as np
import pandas as pd
import time
from math import log
from sys import argv
from ccdc import conformer
from Bio.PDB import PDBList
from ccdc.docking import Docker
from ccdc.io import MoleculeReader, MoleculeWriter
from ccdc.protein import Protein
from ccdc.entry import Entry
from ccdc.molecule import Molecule
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from chembl_webresource_client.new_client import new_client 

### Molecules preparation function

njobs = int(argv[4])
warnings.filterwarnings("ignore")

### Select the path 
parentDir = os.getcwd()

def chembl_mols(chembl_id):
    """
    Download all molecules related with the specific target in chembl, selecting only the unique compunds with 
    activities reported in "IC50","Ki","EC50","Kd" and each molecular weight is between 180 and 900 daltons

    Parameters
    ----------
    chembl_id: str
        ChEMBL id code for a specific target
    
    Return
    ------
    dfActivities: Pandas dataframe
        A dataframe with molecules and other information related with this compunds
    targetName: str
        The common name reported in ChEMBL        
    """
    activity =  new_client.activity
    target = new_client.target
    listOfActivities = activity.filter(target_chembl_id=chembl_id).filter(standard_units="nM").only(
                "canonical_smiles", "molecule_chembl_id", "pchembl_value", 
                "standard_units", "standard_value", "standard_type"
            )
    if not listOfActivities:
        print("Error in ChEMBL ID")
    else:

        targetInfo = target.filter(target_chembl_id=chembl_id).only(
                    "pref_name", "target_type"
                    )
        targetData = pd.DataFrame(targetInfo)
        targetName = targetData["pref_name"][0] 

        ### Do a dataframe and calculate pIC50
        dfActivities = pd.DataFrame(listOfActivities)
        dfActivities = dfActivities[dfActivities['standard_type'].isin(["IC50","Ki","EC50","Kd"])]
        dfActivities = dfActivities.astype({"standard_value":float})
        dfActivities = dfActivities[dfActivities["standard_value"] > 0]
        dfActivities["pIC\u2085\u2080"] = [-log(i / 10**9, 10) for i in dfActivities["standard_value"]]

        ### Sort by pIC50 and delete the repeated molecules

        dfActivities = dfActivities.sort_values(by=["pIC\u2085\u2080"], ascending=False)
        dfActivities = dfActivities.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
        dfActivities = dfActivities[dfActivities['pIC\u2085\u2080'] >= 5]
        
        ### Select molecules only for molecular weight required
        
        mol_weight = []
        for i in dfActivities["canonical_smiles"]:
            mol = []
            try:
                mol.append(Chem.MolFromSmiles(i))
                for j in mol:
                    mol_weight.append(Descriptors.MolWt(j))
            except:
                mol_weight.append(0)
        dfActivities["mol_weight"] = mol_weight
        dfActivities = dfActivities.loc[(dfActivities["mol_weight"] >= 180.0) &
                                        (dfActivities["mol_weight"] <= 600.0), 
                                        ["canonical_smiles", "molecule_chembl_id", "pchembl_value", 
                                        "standard_units", "standard_value", "standard_type",
                                        "mol_weight","pIC\u2085\u2080"]]


        return dfActivities, targetName 

def prep_ligand_from_smiles(smiles, id, dir):
    """
    Prepare molecules for docking using GOLD ligandPreparation function from a smiles

    Parameters
    ----------
    smiles: str
        Smiles to prepare
    id: str
        Name or id for each molecule
    dir: str
        Name of directory to save the file
    
    Return
    ------
    prep_lig: mol2 file
        Molecular file for the prepared structure
    """
    ### molecule from smiles
    lig_molecule = Molecule.from_string(smiles, format="smiles")
    ### Pass ligands to molecule format for GOLD, generating 3D coordinates
    con_gen = conformer.ConformerGenerator()
    con_gen.settings.max_conformers = 1
    lig_mol_3d = con_gen.generate(lig_molecule)
    ligand_prep = Docker.LigandPreparation()
    ligand_prep.settings.protonate = True
    ligand_prep.settings.standardise_bond_types = True
    prep_lig = ligand_prep.prepare(Entry.from_molecule(lig_mol_3d[0].molecule))
    ### Write molecule 
    with MoleculeWriter(f"{dir}/{id}.mol2") as mol_writer:
            mol_writer.write(prep_lig.molecule)
    return prep_lig

### Config file function

def gold_config(protein, ref_ligand, gold_name = "gold", size = 8):
    """
    GOLD configuration file for molecular docking

    Parameters
    ----------
    protein: pdb or mol2 file
        Protein file for make the docking
    ref_ligand: pdb or mol2 file
        Reference ligand for molecular docking
    gold_name: str
        Name for GOLD config file
    size: float
        size of the grid for docking

    Return
    ------
    Gold_config: config file
        GOLD docking config file for make docking
    """        
    ### Change the directory and work here
    os.chdir(path)

    ### call the functions to dock
    docker = Docker()
    settings = docker.settings
    
    ### call protein and native ligand for select the binding site
    prot_1 = protein
    settings.add_protein_file(prot_1)
    native_ligand = ref_ligand
    native_ligand_mol = MoleculeReader(native_ligand)[0]
    prot_dock = settings.proteins[0]

    ### Select parameters to dock: binding site, fitness_function, autoscale, and others
    settings.binding_site = settings.BindingSiteFromLigand(prot_dock, native_ligand_mol, size)
    settings.fitness_function = "PLP"
    settings.autoscale = 200
    settings.early_termination = False
    settings.write_options = "NO_GOLD_LOG_FILE NO_LOG_FILES NO_LINK_FILES NO_RNK_FILES NO_BESTRANKING_LST_FILE NO_GOLD_PROTEIN_MOL2_FILE NO_LGFNAME_FILE NO_PID_FILE NO_SEED_LOG_FILE NO_GOLD_ERR_FILE NO_FIT_PTS_FILES NO_GOLD_LIGAND_MOL2_FILE"
    settings.flip_amide_bonds = True
    settings.flip_pyramidal_nitrogen = True
    settings.flip_free_corners = True
    settings.save_binding_site_atoms = True 
    settings.diverse_solutions = (True, 1, 1.5)

    ### save the configuration file to modify
    Docker.Settings.write(settings,f"{gold_name}.conf")

    ### Add to config file "per_atom_scores"
    with open(f"{gold_name}.conf", "r") as inFile:
        text = inFile.readlines()

    with open(f"{gold_name}.conf", "w") as outFile:
        for line in text:
            if line == "  SAVE OPTIONS\n":
                line = line + "per_atom_scores = 1\n"
            outFile.write(line)

    with open(f"{gold_name}.conf", "r") as inFile:
        text = inFile.readlines()

    with open(f"{gold_name}.conf", "w") as outFile:
        for line in text:
            if line == "  SAVE OPTIONS\n":
                line = line + "concatenated_output = ligand.sdf\n"
            outFile.write(line)

    with open(f"{gold_name}.conf", "r") as inFile:
        text = inFile.readlines()

    with open(f"{gold_name}.conf", "w") as outFile:
        for line in text:
            if line == "concatenated_output = ligand.sdf\n":
                line = line + "output_file_format = MACCS\n"
            outFile.write(line)

### Docking function

def dock_mol(confFile, id, dir, num_sln = 10):
    """
    Docking function 

    Parameters
    ----------
    confFile: str
        Name of the GOLD config file
    id: str
        Name or id for each molecule
    dir: str
        Name of the folder for save dockings
    num_sln: int
        Number of docking solutions
    
    Return
    ------
    docking_sln_file: sdf file 
        Docking solution file in sdf
    """
    conf = confFile
    settings = Docker.Settings.from_file(conf)
    ligand = f"{id}.mol2"
    settings.add_ligand_file(ligand, num_sln)
    settings.output_directory = dir
    settings.output_file = f"{id}_sln.sdf"
    docker = Docker(settings = settings).dock(f"{dir}/{id}.conf")
    return docker

def gen_scaffold(smiles):
    """
    Create a Generic scaffold from smiles

    Parameters
    ----------
    smiles: str
        Smiles to extract generic scaffold
    
    Return
    ------
    scf: str
        Smiles for each scaffold, or errors in this calculation
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            return "Error-Smiles"
        else:
            scf = MakeScaffoldGeneric(mol)
            return Chem.MolToSmiles(scf)
    except:
        return "Something else"

"""
Starting calculations
"""

targets = pd.read_parquet('files/targets_information.parquet')

for row in targets.iloc:
    ### time
    start_cpu = time.process_time()
    start_clock = time.time()

    ### Download the molecules and create datframes from ChEMBL
    dfActivities, targetName = chembl_mols(row['chembl_id'])

    ### Delete bad strings in target name
    unaceptable_strings = ["/", "(", ")", ",", ";", ".", " "]
    for string in unaceptable_strings:
        targetName = targetName.replace(string, "_")
        
    print(
        f'''    
        ********************************************************************************
        
        Starting to work with {targetName}
        
        ********************************************************************************
        '''
    )
    ### Docking types
    docking_types = ['div', 'dcm', 'znc']
                
    ### Protein preparation
    for type in docking_types:
        path = os.path.join('files/', 'docking_sln')
        os.makedirs(path, exist_ok=True)
        prot_name = targetName

        ### Open and prepare protein 
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(row['pdb_id'], file_format='pdb', pdir='files/pdb_files')
        prot = Protein.from_file(f'files/pdb_files/{row['pdb_id']}')
        prot.remove_all_waters()

        ### Split and select one unity if the protein is a homodimer
        ### Select only one chain
        if argv[6] == 'yes':
            if len(prot.chains) >= 2:
                chain = argv[7]
                bad_chain = [c for c in [val.identifier for val in prot.chains] if c not in chain]
                for id in bad_chain:
                    prot.remove_chain(id)

                for lig in prot.ligands:
                    if lig.identifier.split(':')[0] in bad_chain:
                        prot.remove_ligand(lig.identifier)
                        
                for cofactor in prot.cofactors:
                    if cofactor.identifier.split(':')[0] in bad_chain:
                        prot.remove_cofactor(cofactor.identifier)        

        ### Save the principal ligand
        for lig in prot.ligands:
            if lig.identifier.split(':')[1][:3] == argv[3]:
                with MoleculeWriter(path+f"/Ligand_{prot_name}.mol2") as mol_writer:
                    mol_writer.write(lig)        

        ### Remove the ligands and add hydrogens
        for l in prot.ligands:    
            prot.remove_ligand(l.identifier)
        prot.add_hydrogens()

        ### save the protein
        with MoleculeWriter(path+f"/{prot_name}_prep.mol2") as proteinWriter:
            proteinWriter.write(prot)

        ### Create a dataframe with ligands
        dfActivities["rep_num"] = dfActivities.groupby("molecule_chembl_id").cumcount()+1
        dfActivities["id"] = dfActivities["molecule_chembl_id"].astype(str) + "-" + dfActivities["rep_num"].astype(str)
        ligands = dfActivities[["id", "canonical_smiles"]]

        ### Create temporary folder to work
        sysTemp = tempfile.gettempdir()
        myTemp = os.path.join(sysTemp,'mytemp')
        #You must make sure myTemp exists
        if not os.path.exists(myTemp):
            os.makedirs(myTemp)
        
        if type == 'div':
            #now make your temporary sub folder
            act_dir = tempfile.mkdtemp(suffix=None,prefix='act_',dir=myTemp)

            ### Do the preparation in parallel
            if __name__ == "__main__":
                from multiprocessing import Pool
                pool = Pool(processes=njobs)
                for row in ligands.iloc:
                    pool.apply_async(prep_ligand_from_smiles, (row["canonical_smiles"], row["id"], act_dir))
                pool.close()
                pool.join()

            ### Do for the first ligand and protein
            gold_config(f"{path}/{prot_name}_prep.mol2", f"{path}/Ligand_{prot_name}.mol2")

            ### paralelize the docking for use all procesors 
            if __name__ == "__main__":
                from multiprocessing import Pool
                pool = Pool(processes=njobs)
                for row in ligands.iloc:
                    pool.apply_async(dock_mol, ("gold.conf", row["id"], act_dir, 20))
                pool.close()
                pool.join()


            ### Create active and inactive directories and save docking files
            act_f = os.path.join(path, "actives")
            os.makedirs(act_f, exist_ok=True)

            for filename in glob.glob(act_dir + f"/*_sln.sdf"):
                shutil.copy(filename, act_f)

            ### Copy plp_protein to principal folder
            shutil.copy(f"{act_dir}/plp_protein.mol2", path)

            ### Delete dataframes from memory
            gc.collect()

            ### Delete Temporary folders
            shutil.rmtree(myTemp)

            print(
                f'''    
                ********************************************************************************
                
                Time CPU: {round(time.process_time() - start_cpu, 4)}
                Time clock: {round((time.time() - start_clock)/60,4)} min
                
                ********************************************************************************
                '''
            )
        else:       
            ### Calculate the scaffolds in parallel
            if __name__=='__main__':
                from multiprocessing import Pool
                with Pool() as pool:
                    scfs = pool.map(gen_scaffold, ligands["canonical_smiles"].tolist())

            ### add to ligands dataframe the column with the scaffolds and delete the errors in this calculation
            ligands["scf"] = scfs
            ligands = ligands[ligands["scf"] != "Error-Smiles"]
            ligands = ligands[ligands["scf"] != "Something else"]

            ### Decoys database
            if type == 'znc':
                dec = pd.read_parquet("files/znc_scf.parquet", sep=",")
            elif type == 'dcm':
                dec = pd.read_parquet('files/DCM_prepared.parquet', sep=',')  

            ### Create a list with ids between ligands_1 and Zinc and delete the rows in dataframe
            bad_id_dec = pd.merge(dec, ligands, on="scf")["id_x"].unique().tolist()
            dec_filtered = dec[~dec.id.isin(bad_id_dec)]

            ### Select 4 times te total of active molecules from filtered Zinc
            list_index = random.sample(range(len(dec_filtered)), len(ligands)*4)
            inactives = dec.iloc[list_index]

            #now make your temporary sub folder
            ina_dir = tempfile.mkdtemp(suffix=None,prefix='ina_',dir=myTemp)

            ### Do the preparation in parallel
            if __name__ == "__main__":
                from multiprocessing import Pool
                pool = Pool(processes=njobs)
                for row in inactives.iloc:
                    pool.apply_async(prep_ligand_from_smiles, (row["smiles"], row["id"], ina_dir))
                pool.close()
                pool.join()

            ### Do for the first ligand and protein
            gold_config(f"{path}/{prot_name}_prep.mol2", f"{path}/Ligand_{prot_name}.mol2")

            ### paralelize the docking for use all procesors 
            if __name__ == "__main__":
                from multiprocessing import Pool
                pool = Pool(processes=njobs)
                for row in inactives.iloc:
                    pool.apply_async(dock_mol, ("gold.conf", row["id"], ina_dir, 20))
                pool.close()
                pool.join()


            ### Create active and inactive directories and save docking files
            ina_f = os.path.join(path, "Decoys")
            os.makedirs(ina_f, exist_ok=True)

            for filename in glob.glob(ina_dir + f"/*_sln.sdf"):
                shutil.copy(filename, ina_f)

            ### Copy plp_protein to principal folder
            shutil.copy(f"{ina_dir}/plp_protein.mol2", path)

            ### Delete dataframes from memory
            gc.collect()

            ### Delete Temporary folders
            shutil.rmtree(myTemp)

            print(
                f'''    
                ********************************************************************************
                
                Time CPU: {round(time.process_time() - start_cpu, 4)}
                Time clock: {round((time.time() - start_clock)/60,4)} min
                
                ********************************************************************************
                '''
            )