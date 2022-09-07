"""
Workflow to extract PADIF, PADIF2, PROLIF, ECIF from GOLD docking results for train machine learning models

Felipe Victoria-Munoz

Parameters
----------
argv[1]: dir
    Directory with active and inactive folders and results from gold docking
argv[2]: dir
    Directory for save the protein ligand interaction fingerprints

Return
------
PADIF: csv 
    Table with PADIF fingerprint in a csv file
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
from sys import argv

### Useful functions
### Extract chemplp from files function

def chemplp_ext(file):
    """
    Function for open docking solution file and extract protein score contributions

    Parameters
    ----------
    file: mol2 or sdf file
        File of the docking solutions

    Return
    lista: list
        List of contributions per each atom
    """    
    lista = []
    limits = []
    with open(file) as inFile:  
        for num, line in enumerate(inFile):
            if "> <Gold.PLP.Protein.Score.Contributions>\n" in line:
                limits.append(num)
            if "$$$$\n" in line:
                limits.append(num)
    with open(file) as inFile:     
        for num, line in enumerate(inFile):            
            if num > limits[0] and num < limits[1]-1:
                lista.append(line)
    return lista

### Sign change without affect zeros

def sign_change(list):
    """
    Function for change the sign of chemplp contributions, avoiding the change in zeros

    Parameters
    ----------
    list: list
        List of chemplp contributions
    
    Return
    ------
    new_list: list
        List of chemplp contributions with the sign changed
    """
    new_list = []
    for value in list:
        if value > 0:
            x = value*-1
        else:
            x = value
        new_list.append(x)
    return new_list

### Organize chemplp values in Dataframes

def chemplp_dataframe(soln_files_names, etiquete, dir):
    """
    Extract chemplp for multiple docking solutions files

    Parameters
    ----------
    soln_files_names: str
        Names of docking solution files for processing
    etiquete: str
        String to indetify the class of molecules
    dir: str or path
        Directory of files
    
    Return
    ------
    df: pandas dataframe
        Dataframe with all interactions 
    """  
    chempl_list = []
    ids =  []
    for name in glob.glob(soln_files_names):
        chempl_list.append(chemplp_ext(name))
        val = name.strip(f"/{dir}/")
        ids.append(val.split("_")[0])

    ### Organize this list and create list of list
    list_0 = []
    for list in chempl_list: 
        list_1 = []
        for row in list:
            list_2 = []
            for value in row.split():
                list_2.append(value)
            list_1.append(list_2)
        list_0.append(list_1)

    ### Pass to Dataframe
    df = []
    columns = []
    for lista in list_0:
        df.append(pd.DataFrame(lista))
        columns.append(lista[0])

    df = pd.concat(df)
    df.columns = columns[0]

    ### Create a identifier of each molecule  
    counter = 0
    list_of_mols = [None]*len(df)
    for idx, row in enumerate(df["AtomID"]):
        if row == "AtomID":
            counter +=1
        list_of_mols[idx] = etiquete + "-" + str(counter)
    df["RunID"] = list_of_mols
    
    ### Delete the unuseful rows
    df = df.dropna()
    df = df[df["AtomID"] != "AtomID"]
    df = df.drop(columns=["PLP.total"])

    ### Change the type of data of values and sort by molecules
    df = df.astype({"ChemScore_PLP.Hbond": float, "ChemScore_PLP.CHO": float,
                    "ChemScore_PLP.Metal": float, "PLP.S(hbond)": float,
                    "PLP.S(metal)": float, "PLP.S(buried)": float,
                    "PLP.S(nonpolar)": float, "PLP.S(repulsive)": float})
    df = df.sort_values(by = ["RunID"], ignore_index= True)

    ### change the value for the fisrt 3 columns
    df["ChemScore_PLP.Hbond"] = sign_change(df["ChemScore_PLP.Hbond"])
    df["ChemScore_PLP.CHO"] = sign_change(df["ChemScore_PLP.Hbond"])
    df["ChemScore_PLP.Metal"] = sign_change(df["ChemScore_PLP.Hbond"])

    return df 

### From chemplp datframes generate PADIF 

def padif_gen(active_df, inactive_df):
    """
    Create a unique PADIF, with interactions between actives and inactive datasets

    parameters
    ----------
    active_df: pandas dataframe
        Dataframe with chemplp of actives compounds
    inactive_df: pandas dataframe
        Dataframe with chemplp of inactives compunds
    
    return
    ------
    padif: pandas dataframe
        Dataframe with mix of interactions in the actives and inactives datasets
    """ 
    ### Join the dataframes and do the chemplp
    df_tot = pd.concat([active_df, inactive_df])

    ### Do the PADIF for each molecule and organize in a dataframe
    padif = df_tot.pivot(index="RunID", columns="AtomID")
    padif.columns = padif.columns.map('{0[0]}_{0[1]}'.format)
    padif.index = padif.index.tolist()
    padif.fillna(0.0, inplace= True)
    padif = padif.loc[:,(padif.sum(axis=0) != 0)]

    ### Add column to active or inactive classification
    Act_type = []
    for value in padif.index.tolist():
        if value[:3] == "Act":
            Act_type.append("Active")
        else:
            Act_type.append("Inactive")
    padif["Activity"] = Act_type

    return padif

def chemplp_2(plp_protein_file, chemplp_dataframe):
    """
    This function extract for plp_protein file the list of atoms and its clasffication for build PADIF+

    parameters
    ----------
    plp_proteim_file: file
        File from docking process that conteins info from protein
    chemplp_dataframe: pandas dataframe
        Dataframe with chemplp contributions of protein
    
    return:
    -------
    protein_atoms: pandas dataframe
        Dataframe with atom type and residue for each atom in protein
    """
    list = []
    limits = []
    with open(plp_protein_file) as inFile:   
        for num, line in enumerate(inFile):
            if "@<TRIPOS>ATOM\n" in line:
                limits.append(num)
            if "@<TRIPOS>BOND\n" in line:
                limits.append(num)
                
    with open(plp_protein_file) as inFile: 
        for num, line in enumerate(inFile):
            if num > limits[0] and num < limits[1]-1:
                    list.append(line)
    
    ### Generate list of list
    list_2 = []
    for row in list:
        list_3 = []       
        for value in row.split():
            list_3.append(value)
        list_2.append(list_3)

    ### Create the dataframe
    columns = ["atom_id", "atom", "x", "y", "z", "atom_type", "sub_id", "residue", "charge"]
    protein_atoms = pd.DataFrame(data = list_2, columns = columns)
    mask = protein_atoms[["charge"]].isna().all(axis=1)
    protein_atoms.loc[mask, "atom_type":"charge"] = protein_atoms.loc[mask, "atom_type":"charge"].shift(1, axis=1)  
    protein_atoms = protein_atoms.set_index("atom_id")
    protein_atoms.index = protein_atoms.index.tolist()

    ### Classify and add to table atom_type and res_type for see the behavior
    atoms_id = [val-1 for val in map(int, chemplp_dataframe["AtomID"].unique().tolist())]
    df1 = protein_atoms.iloc[atoms_id]

    ### Split the residue rwo into res and number
    df1["res"] = df1["residue"].str.slice(stop=3)
    df1["res_num"] = df1["residue"].str.slice(start=3)

    ### Create list to classify residues
    residue_type = []

    for value in df1["res"]:
        Hydrophobic_Aliphatic = ["GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "MET"]
        Hydrophobic_Aromatic = ["PHE", "TYR", "TRP"]
        Hydrophilic_Polar_uncharged = ["SER", "THR", "CYS", "ASN", "GLN"]
        Hydrophilic_Acidic = ["ASP", "GLU"]
        Hydrophilic_Basic = ["ARG", "HIS", "LYS"]
        
        if value in Hydrophobic_Aliphatic:
            residue_type.append(1)
        
        elif value in Hydrophobic_Aromatic:
            residue_type.append(2)
        
        elif value in Hydrophilic_Polar_uncharged:
            residue_type.append(3)
        
        elif value in Hydrophilic_Acidic:
            residue_type.append(4)
        
        elif value in Hydrophilic_Basic:
            residue_type.append(5)
        else:
            residue_type.append(0)    

    df1["residue_type"] = residue_type
    df1["AtomID"] = df1.index
    df = chemplp_dataframe.merge(df1[["AtomID","residue_type"]], how="left")

    ### Clasify for type of atom
    atom_type =[]

    for value in df1["atom_type"]:
        if value == "NONPOLAR":
            atom_type.append(1)
        
        if value == "ACCEPTOR":
            atom_type.append(2)
        
        if value == "DONOR":
            atom_type.append(3)
        
        if value == "COORD":
            atom_type.append(4)
        
        if value == "DONACC":
            atom_type.append(5)
        
        if value == "METAL":
            atom_type.append(6)
        
    df1["atom_type"] = atom_type
    df = df.merge(df1[["AtomID","atom_type"]], how="left")

    return df

### Avoid unuseful warnings 
warnings.filterwarnings("ignore")

### Name for create a new folder / Its the name of the last folder 
os.chdir(argv[1])
parentDir = os.getcwd()
targetName = str(argv[1]).split("/")[-2]

### Create a new directory for save the diferents plifs
folder = argv[2]
path = os.path.join(folder, targetName)
os.makedirs(path, exist_ok=True)

### Open and create PADIF from active and inactive folder

### Actives
act_dir = [x[0] for x in os.walk(parentDir)][1]
act_df = chemplp_dataframe(f"{act_dir}/*_sln.sdf", "Act", act_dir)

### Inactives
ina_dir = [x[0] for x in os.walk(parentDir)][2]
ina_df = chemplp_dataframe(f"{ina_dir}/*_sln.sdf", "Ina", ina_dir)

### Generate and save PADIF
padif = padif_gen(act_df, ina_df)
padif.to_csv(f"{path}/{targetName}_PADIF.csv", sep= ",")

### Create a dataframe of chemplp2, with atom and residue types
act_chemplp_2 = chemplp_2(f"{parentDir}/plp_protein.mol2", act_df)
ina_chemplp_2 = chemplp_2(f"{parentDir}/plp_protein.mol2", ina_df)

### Generate and save PADIF
padif2 = padif_gen(act_chemplp_2, ina_chemplp_2)
padif2.to_csv(f"{path}/{targetName}_PADIF2.csv", sep= ",")
