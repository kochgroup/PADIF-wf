"""
Workflow to extract PADIF, PADIF2, PROLIF, ECIF from GOLD docking results for train machine learning models

Felipe Victoria-Muñoz

Parameters
----------
argv[1]: dir
    Directory with active and inactive folders and results from gold docking
argv[2]: dir
    Directory for save the protein ligand interaction fingerprints

Return
------
PADIF: csv file 
    Table with PADIF fingerprint in a csv file - ref "10.1186/s13321-018-0264-0"
PADIF2: csv file
    Table with PADIF2 (PADIF + Residues and atoms classifications) fingerprint in a csv file
PROLIF: csv file
    Table with PROLIF fingerprint in a csv file - ref "10.1186/s13321-021-00548-6"
ECIF: csv file
    Table with ECIF fingerprint in a csv file - ref "10.1093/bioinformatics/btaa982"
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
import prolif as plf
from sys import argv
import prolif as plf
from tqdm import tqdm
import MDAnalysis as mda
from rdkit import RDLogger
from os import listdir
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

### Avoid warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

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
    name = dir.split("/")[-1]
    for name in tqdm(glob.glob(soln_files_names), desc=f"PADIF for {name} molecules"):
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

    ### Get Dataframe
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
    padif["activity"] = Act_type

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

### PROLIF calculation

def unique(list1):
    """
    Create a list with unique values

    Parameters
    ----------
    list1: list
        List to process
    
    Return
    ------
    un_list: list
        list with unique values
    """
    x = np.array(list1)
    un_list = np.unique(x).tolist()
    return un_list

def prolif_ext(sdf_file, protein_prolif):
    """
    Function for extract PROLIF fingerprint from sdf docking solition files per each atom of protein

    Parameters
    ----------
    sdf_file: str or path
        SDF docking solution file
    protein_prolif: pdb file
        Protein prolif variable
    
    Return
    ------
    df: pandas dataframe
        Dataframe with columns per each atom of protein
    """
    
    # load ligands
    path_1 = sdf_file
    try:
        lig_suppl = plf.sdf_supplier(path_1)
        # generate fingerprint
        fp = plf.Fingerprint()
        fp.run_from_iterable(lig_suppl, protein_prolif, progress=False)
        df = fp.to_dataframe(dtype=np.uint8)
        df = df.reindex(sorted(df.columns), axis=1)
        ### Select only the best conformation and delete de null rows
        df = df[:1]
        ### Join columns names between residue and type of interaction 
        df.columns = df.columns.droplevel()
        df.columns = df.columns.map('{0[0]}_{0[1]}'.format)
        ### Search per atom the of each residue
        df1 = fp.to_dataframe(return_atoms=True)
        df1 = df1.T
        ### List of atoms
        l1 = []
        for lista in df1.values:
            l = []
            for group in lista:
                if group[1] != None:
                    l.append(group[1])
            l1.append(unique(l))
        ### Unique atoms
        atom_unq = []
        mult_atom_indx = []
        for idx, lista in enumerate(l1):
            if len(lista) > 1:
                mult_atom_indx.append(idx)
            else:
                atom_unq.append(lista[0])
        ### Add atoms to columns
        ### If exist multple interactions per atom
        if mult_atom_indx != []:
            ### Extract columns with one interaction per atom
            df = df.drop(df.columns[mult_atom_indx], axis=1)
            col_ma = [x for x in df.columns.tolist() if x not in df.columns.tolist()]
            atom_unq_s = [str(val) for val in atom_unq]
            list_col = df.columns.tolist()
            df.columns = [s1 + "_"  + s2  for s1, s2 in zip(list_col, atom_unq_s)]
            ### Delete the columns without interactions
            df = df.loc[:, (df != 0).any(axis=0)]
        else:
            list_col = df.columns.tolist()
            df.columns = [s1 + "_"  + s2  for s1, s2 in zip(list_col, atom_unq)]
            ### Delete the columns without interactions
            df = df.loc[:, (df != 0).any(axis=0)]
        return df 
    except:
        pass

def prolif_for_ML(folder, protein, types_mols=["Actives", "Inactives"]):
    """
    Function to get sdf docking results to dataframe for ML classification

    Parameters
    ----------
    path: str
        path or directory of docking results
    protein: pdb file
        PDB protein file used in molecular docking
    types_mols:
        Names of folders where are actives and inactives molecules

    Return:
    prolif: pandas dataframe
        Dataframe with PROLIF intereactions classified by active and inactive
    """
    # load protein
    prot = mda.Universe(protein)
    prot = plf.Molecule.from_mda(prot)

    ### Extract a list for each type of activities
    prolif_lst = []
    for types in types_mols:
        prolif= []
        for name in tqdm(glob.glob(f"{folder}/{types}/*_sln.sdf"), desc=f"PROLIF for {types} molecules"):
            prolif.append(prolif_ext(name, prot))

        ### Join all molecules in one list and add the index by activity
        act = pd.concat(prolif)
        act["activity"] = types[:-1]
        act.index = [types[:3] + "_" + str(num+1) for num in range(len(act))]
        act = act.fillna(0.0)
        prolif_lst.append(act)

    ### Join actives and inactives datafremes in one dataframe and put in the last postiion activity column
    prolif = pd.concat(prolif_lst)
    prolif = prolif.fillna(0.0)
    new_cols = [col for col in prolif.columns if col != "activity"] + ["activity"]
    prolif = prolif[new_cols]
    
    return prolif

### ECIF calculation

# Possible predefined protein atoms
ECIF_ProteinAtoms = ['C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1',
                     'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1',
                     'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0',
                     'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0']

# Possible ligand atoms according to the PDBbind 2016 "refined set"
ECIF_LigandAtoms = ['Br;1;1;0;0;0', 'C;3;3;0;1;1', 'C;4;1;1;0;0', 'C;4;1;2;0;0',
                     'C;4;1;3;0;0', 'C;4;2;0;0;0', 'C;4;2;1;0;0', 'C;4;2;1;0;1',
                     'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1', 'C;4;3;0;0;0',
                     'C;4;3;0;0;1', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;4;4;0;0;0', 'C;4;4;0;0;1', 'C;5;3;0;0;0', 'C;5;3;0;1;1',
                     'C;6;3;0;0;0', 'Cl;1;1;0;0;0', 'F;1;1;0;0;0', 'I;1;1;0;0;0',
                     'N;3;1;0;0;0', 'N;3;1;1;0;0', 'N;3;1;2;0;0', 'N;3;2;0;0;0',
                     'N;3;2;0;0;1', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;0;1',
                     'N;3;2;1;1;1', 'N;3;3;0;0;0', 'N;3;3;0;0;1', 'N;3;3;0;1;1',
                     'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;1;0;0', 'N;4;2;2;0;0',
                     'N;4;2;2;0;1', 'N;4;3;0;0;0', 'N;4;3;0;0;1', 'N;4;3;1;0;0',
                     'N;4;3;1;0;1', 'N;4;4;0;0;0', 'N;4;4;0;0;1', 'N;5;2;0;0;0',
                     'N;5;3;0;0;0', 'N;5;3;0;1;1', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'O;2;2;0;0;0', 'O;2;2;0;0;1', 'O;2;2;0;1;1', 'P;5;4;0;0;0',
                     'P;6;4;0;0;0', 'P;6;4;0;0;1', 'P;7;4;0;0;0', 'S;2;1;0;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0', 'S;2;2;0;0;1', 'S;2;2;0;1;1',
                     'S;3;3;0;0;0', 'S;3;3;0;0;1', 'S;4;3;0;0;0', 'S;6;4;0;0;0',
                     'S;6;4;0;0;1', 'S;7;4;0;0;0', 'B;3;3;0;0;0', 'B;3;3;0;0;1']

PossibleECIF = [i[0]+"-"+i[1] for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]

# Atom keys
parentDir = os.getcwd()
ak_file = parentDir + "/files/PDB_Atom_Keys.csv"
Atom_Keys=pd.read_csv(ak_file, sep=",")

def GetAtomType(atom):
# This function takes an atom in a molecule and returns its type as defined for ECIF
    
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())), 
               ]

    return(";".join(AtomType))

def LoadSDFasDF(SDF):
# This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF
    
    m = Chem.MolFromMolFile(SDF, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    
    ECIF_atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(GetAtomType(atom))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            ECIF_atoms.append(entry)

    df = pd.DataFrame(ECIF_atoms)
    df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE","X","Y","Z"]
    
    return(df)

def LoadPDBasDF(PDB):
# This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF

    ECIF_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                ECIF_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)

def GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=6.0):
# This function returns the protein-ligand atom-type pairs for a given distance cutoff
    
    # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)
    
    # Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    # Get all possible pairs
    Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    Pairs = pd.concat([Pairs,Distances], axis=1)
    Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    # Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
    Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
    return Pairs

def GetECIF(PDB_protein, SDF_ligand, distance_cutoff=6.0):
# Main function for the calculation of ECIF
    Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
    ECIF = [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]
    return ECIF

def ecif_to_ML(folder, protein, types_mols = ["Actives", "Inactives"]):
    """
    Get dataframe with ecif fingeprint for machine learnign

    Parameters
    ----------
    path: str
        path or directory of docking results
    protein: pdb file
        PDB protein file used in molecular docking
    types_mols:
        Names of folders where are actives and inactives molecules

    Return:
    ecif: pandas dataframe
        Dataframe with ECIF intereactions classified by active and inactive
    """
    ecif_lst = []
    for types in types_mols:
        ecif_l = []
        for name in tqdm(glob.glob(f"{folder}/{types}/*_sln.sdf"), desc=f"ECIF for {types} molecules"):
            ecif = GetECIF(protein , name)
            ecif_l.append(ecif)    

        ### Join all molecules in one list and add the index by activity
        cols = [f"Vector_{n+1}" for n in range(1584)]
        act = pd.DataFrame(ecif_l, columns= cols) 
        act["activity"] = types[:-1]
        act.index = [types[:3] + "_" + str(num+1) for num in range(len(act))]
        ecif_lst.append(act)

    ecif = pd.concat(ecif_lst)
    
    return ecif

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
print(f"PADIF and PADIF2 made for {targetName}")

### Convert protein mol2 file to pdb file
prot_mol2 = mda.Universe(f"{parentDir}/{targetName}_prep.mol2")
with mda.Writer(f"{path}/{targetName}.pdb") as pdb:
    pdb.write(prot_mol2)

### PROLIF from results
prolif = prolif_for_ML(argv[1], f"{path}/{targetName}.pdb")
prolif.to_csv(f"{path}/{targetName}_PROLIF.csv", sep= ",")

### ECIF from results
ecif = ecif_to_ML(argv[1], f"{path}/{targetName}.pdb")
ecif.to_csv(f"{path}/{targetName}_ECIF.csv", sep= ",")

### Remove Protein file
os.remove(f"{path}/{targetName}.pdb")