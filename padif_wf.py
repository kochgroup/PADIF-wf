"""
Workflow to do PADIF 
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
from math import log
from sys import argv
from ccdc import conformer
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

njobs = int(argv[3])
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dcm = pd.read_csv("/nfs/home/dvictori/Documents/DCM/DCM_prepared.csv", sep=",")

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
                                            (dfActivities["mol_weight"] <= 900.0), 
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
        ligand_prep.standardise_bond_types = True
        prep_lig = ligand_prep.prepare(Entry.from_molecule(lig_mol_3d[0].molecule))
        ### Write molecule 
        with MoleculeWriter(f"{dir}/{id}.mol2") as mol_writer:
                mol_writer.write(prep_lig.molecule)
        return prep_lig

    ### Config file function

    def gold_config(protein, ref_ligand, gold_name = "gold", size = 15.0):
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

    def dock_mol(confFile, id, dir, num_sln = 100):
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

    ### Generic scaffolds function

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
    Starting the calculations
    """

    ### Download the molecules and create datframes from ChEMBL
    dfActivities, targetName = chembl_mols(argv[1])

    ### Delete bad strings in target name
    unaceptable_strings = ["/", "(", ")", ",", ";", ".", " "]
    for string in unaceptable_strings:
        targetName = targetName.replace(string, "_")
                
    ### Protein preparation
    ### Select the path
    parentDir = os.getcwd()
    path = os.path.join(parentDir, targetName)
    os.makedirs(path)
    prot_name = targetName

    ### Open and prepare protein 
    prot1 = argv[2]
    prot = Protein.from_file(prot1)
    prot.remove_all_waters()

    ### Select only one chain
    if len(prot.chains) >= 2:
        identifiers = []
        for idx, chain in enumerate(prot.chains):
            if idx > 0:
                identifiers.append(chain.identifier)
        for id in identifiers:
            prot.remove_chain(id)

    ### Save the principal ligand
    weight = []
    for lig in prot.ligands:
        weight.append(lig.molecular_weight)
    for lig in prot.ligands:
        if lig.molecular_weight == max(weight):
            with MoleculeWriter(path+f"/Ligand_{prot_name}.mol2") as mol_writer:
                mol_writer.write(lig)

    ### Remove the ligands and add hydrogens
    ligand = prot.ligands
    for l in ligand:
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
            pool.apply_async(dock_mol, ("gold.conf", row["id"], act_dir, 100))
        pool.close()
        pool.join()

    ### Extract chemplp from best docking solutions as actives
    act_df = chemplp_dataframe(f"{act_dir}/*_sln.sdf", "Act", act_dir)

    ### Calculate the scaffolds in parallel
    if __name__=='__main__':
        from multiprocessing import Pool
        with Pool() as pool:
            scfs = pool.map(gen_scaffold, ligands["canonical_smiles"].tolist())

    ### add to ligands dataframe the column with the scaffolds and delete the errors in this calculation
    ligands["scf"] = scfs
    ligands = ligands[ligands["scf"] != "Error-Smiles"]
    ligands = ligands[ligands["scf"] != "Something else"]

    ### Create a list with ids between ligands_1 and Zinc and delete the rows in dataframe
    bad_id_dcm = pd.merge(dcm, ligands, on="scf")["id_x"].unique().tolist()
    dcm_filtered = dcm[~dcm.id.isin(bad_id_dcm)]

    ### Select 4 times te total of active molecules from filtered Zinc
    # random.seed(1234)
    list_index = random.sample(range(len(dcm_filtered)), len(ligands)*4)
    inactives = dcm.iloc[list_index]

    ### Create inactive directory
    ina_dir = tempfile.mkdtemp(suffix=None,prefix='ina_',dir=myTemp)

    ### Do the preparation in parallel
    if __name__ == "__main__":
        from multiprocessing import Pool
        pool = Pool(processes=njobs)
        for row in inactives.iloc:
            pool.apply_async(prep_ligand_from_smiles, (row["smiles"], row["id"], ina_dir))
        pool.close()
        pool.join()

    ### paralelize the docking for use all procesors 
    if __name__ == "__main__":
        from multiprocessing import Pool
        pool = Pool(processes=njobs)
        for row in inactives.iloc:
            pool.apply_async(dock_mol, ("gold.conf", row["id"], ina_dir))
        pool.close()
        pool.join()

    ### Extract chemplp from best docking solutions as actives
    ina_df = chemplp_dataframe(f"{ina_dir}/*_sln.sdf", "Ina", ina_dir)

    ### Do the PADIF_1
    padif_3 = padif_gen(act_df, ina_df)
    padif_3.to_csv(f"{path}/{prot_name}_PADIF.csv", sep =",")

    ### Create active and inactive directories and save docking files
    act_f = os.path.join(path, "Actives")
    os.makedirs(act_f, exist_ok=True)

    for filename in glob.glob(act_dir + f"/*_sln.sdf"):
        shutil.copy(filename, act_f)
    
    ### Create active directory
    ina_f = os.path.join(path, "Inactives")
    os.makedirs(ina_f, exist_ok=True)

    for filename in glob.glob(ina_dir + f"/*_sln.sdf"):
        shutil.copy(filename, ina_f)

    ### Delete dataframes from memory
    gc.collect()

    ### Delete Temporary folders
    shutil.rmtree(myTemp)