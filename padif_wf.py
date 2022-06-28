"""
Workflow to do PADIF
"""
import os
import gc
import glob
import random
import warnings
import numpy as np
import pandas as pd
import modin.pandas as pd_m
from math import log
from sys import argv
from distributed import Client
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
    client = Client()
    znc = pd_m.read_csv("/nfs/home/dvictori/Documents/zinc_fp/znc_scf.csv", sep=",")

    def chembl_mols(chembl_id):
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
            dfActivities = pd_m.DataFrame(listOfActivities)
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

            ### divide data between actives and inactives

            dfActives = dfActivities[:int(np.ceil(len(dfActivities)*0.25))]
            dfInactives = dfActivities[int(np.ceil(len(dfActivities)*.25)):]

            return dfActivities, dfActives, dfInactives, targetName 

    def prep_ligand_from_smiles(smiles, id, dir):
        ### molecule from smiles
        lig_molecule = Molecule.from_string(smiles, format="smiles")
        ### Pass ligands to molecule format for GOLD, generating 3d coordinates
        con_gen = conformer.ConformerGenerator()
        con_gen.settings.max_conformers = 1
        lig_mol_3d = con_gen.generate(lig_molecule)
        ### Prepare entries, I desactivate protonation protocol, beacuse this is uncompatible with protonation_states
        ligand_prep = Docker.LigandPreparation()
        ligand_prep.settings.protonate = True
        ligand_prep.standardise_bond_types = True
        prep_lig = ligand_prep.prepare(Entry.from_molecule(lig_mol_3d[0].molecule))
        ### Write molecule 
        with MoleculeWriter(f"{dir}/{id}.mol2") as mol_writer:
                mol_writer.write(prep_lig.molecule)
        return prep_lig

    ### Config file function

    def gold_config(protein, ref_ligand):        
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
        settings.binding_site = settings.BindingSiteFromLigand(prot_dock, native_ligand_mol, 15.0)
        settings.fitness_function = "PLP"
        settings.autoscale = 200
        settings.early_termination = False
        settings.write_options = "NO_LOG_FILES NO_LINK_FILES NO_RNK_FILES NO_BESTRANKING_LST_FILE NO_GOLD_PROTEIN_MOL2_FILENO_LGFNAME_FILE NO_PID_FILENO_SEED_LOG_FILE NO_GOLD_ERR_FILE NO_FIT_PTS_FILES NO_GOLD_LIGAND_MOL2_FILE"
        settings.flip_amide_bonds = True
        settings.flip_pyramidal_nitrogen = True
        settings.flip_free_corners = True

        ### save the configuration file to modify
        Docker.Settings.write(settings,"gold.conf")

        ### Add to config file "per_atom_scores"
        with open("gold.conf", "r") as inFile:
            text = inFile.readlines()

        with open("gold.conf", "w") as outFile:
            for line in text:
                if line == "  SAVE OPTIONS\n":
                    line = line + "per_atom_scores = 1\n"
                outFile.write(line)
        
        with open("gold.conf", "r") as inFile:
            text = inFile.readlines()

        with open("gold.conf", "w") as outFile:
            for line in text:
                outFile.write(line.replace("make_subdirs = 0\n", "make_subdirs = 1\n"))

    ### Docking function

    def dock_mol(confFile, id, dir):
        conf = confFile
        settings = Docker.Settings.from_file(conf)
        ligand = f"{id}.mol2"
        settings.add_ligand_file(ligand, 100)
        settings.output_directory = dir
        docker = Docker(settings = settings).dock(f"{dir}/{id}.conf")
        return docker

    ### Extract chemplp from files function

    def chemplp_ext(file):    
        lista = []
        with open(file) as inFile:   
            for num, line in enumerate(inFile):
                if "> <Gold.PLP.Protein.Score.Contributions>\n" in line:
                    for num2, line in enumerate(inFile):
                        if num2 < num:
                            lista.append(line)    
        return lista

    ### Sign change without affect zeros
    def sign_change(list):
        new_list = []
        for value in list:
            if value > 0:
                x = value*-1
            else:
                x = value
            new_list.append(x)
        return new_list

    ### Organize chemplp values in Dataframes

    def chemplp_dataframe(soln_files_names, etiquete):  
        chempl_list = []
        ids =  []
        for name in glob.glob(soln_files_names):
            chempl_list.append(chemplp_ext(name))
            val = name.split("gold_soln_")[1]
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
        df = pd.DataFrame()
        for list in list_0:
            columns = list[0]
            df_n = pd.DataFrame(list, columns=columns)
            df =  df.append(df_n)

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
    dfActivities, dfActives, dfInactives, targetName = chembl_mols(argv[1])

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

    ### Open ligands and create dataframe with the most important values
    dfActivities["rep_num"] = dfActivities.groupby("molecule_chembl_id").cumcount()+1
    dfActivities["id"] = dfActivities["molecule_chembl_id"].astype(str) + "-" + dfActivities["rep_num"].astype(str)
    ligands = dfActivities[["id", "canonical_smiles"]]

    ### Create actives directory to work
    act_dir = os.path.join(path, "actives")
    os.makedirs(act_dir, exist_ok=True)

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
            pool.apply_async(dock_mol, ("gold.conf", row["id"], act_dir))
        pool.close()
        pool.join()

    ### Extract chemplp from best docking solutions as actives
    act_df1 = chemplp_dataframe(f"{act_dir}/*/*_1.mol2", "Act")

    ### Extract chemplp from other docking solutions as inactives
    df1 = []
    for num in random.sample(range(2,101), 4):
        df1.append(chemplp_dataframe(f"{act_dir}/*/*_{num}.mol2", f"Ina_{num-1}"))
    ina_df1 = pd.concat(df1)

    ### Do the PADIF_1
    padif_1 = padif_gen(act_df1, ina_df1)
    padif_1.to_csv(f"{path}/{prot_name}_PADIF-1.csv", sep =",")

    ### Split the molecules in chembl between actives and inactives
    ### Open the first quartile of database as actives and create a list with ids

    act_id = dfActives["molecule_chembl_id"].tolist()
    ina_id = dfInactives["molecule_chembl_id"].tolist()

    ### Do actives and inactives dataframe
    df2 = []
    counter = 1
    for id in act_id:
        try:
            df2.append(chemplp_dataframe(f"{act_dir}/{id}*/*_1.mol2", f"Act_{counter}"))
            counter += 1
        except:
            pass
    act_df2 = pd.concat(df2, ignore_index= True)

    df3 = []
    counter = 1
    for id in ina_id:
        try:
            df3.append(chemplp_dataframe(f"{act_dir}/{id}*/*_1.mol2", f"Ina_{counter}"))
            counter += 1
        except:
            pass
    ina_df2 = pd.concat(df3, ignore_index= True)

    ### Do PADIF_2
    padif_2 = padif_gen(act_df2, ina_df2)
    padif_2.to_csv(f"{prot_name}_PADIF-2.csv", sep =",")

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
    bad_id_znc = pd_m.merge(znc, ligands, on="scf")["id_x"].unique().tolist()
    znc_filtered = znc[~znc.id.isin(bad_id_znc)]

    ### Select 4 times te total of active molecules from filtered Zinc
    # random.seed(1234)
    list_index = random.sample(range(len(znc_filtered)), len(ligands)*4)
    inactives = znc.iloc[list_index]

    ### Create inactive directory
    ina_dir = os.path.join(path, "inactives")
    os.makedirs(ina_dir, exist_ok=True)

    ### Do the preparation in parallel
    if __name__ == "__main__":
        from multiprocessing import Pool
        pool = Pool(processes=njobs)
        for row in inactives.iloc:
            pool.apply_async(prep_ligand_from_smiles, (row["std_smiles"], row["id"], ina_dir))
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
    ina_df3 = chemplp_dataframe(f"{ina_dir}/*/*_1.mol2", "Ina")

    ### Do the PADIF_1
    padif_3 = padif_gen(act_df1, ina_df3)
    padif_3.to_csv(f"{path}/{prot_name}_PADIF-3.csv", sep =",")

    ### Delete dataframes
    gc.collect()
