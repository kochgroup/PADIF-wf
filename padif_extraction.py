#%%
"""
Python script for extract PADIF from docking solutions
"""

import os
from glob import glob
import pandas as pd
import random
from tqdm import tqdm
import multiprocessing

def padif_dataframe(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the data into records using the entry separator
    records = data.split('$$$$')

    transformed_tables = []

    # Iterate over each record
    for record in records:
        if '> <Gold.PLP.Protein.Score.Contributions>' in record:
            # Extract table content
            table_start = record.find('> <Gold.PLP.Protein.Score.Contributions>') + len('> <Gold.PLP.Protein.Score.Contributions>\n')
            table_end = record.find('\n\n', table_start)
            table_content = record[table_start:table_end].strip()
            
            if table_content:
                from io import StringIO
                df = pd.read_csv(StringIO(table_content), delim_whitespace=True)
                
                # Pivot the DataFrame
                df_pivot = df.melt(id_vars=['AtomID'], var_name='Metric', value_name='Value')
                df_pivot['Metric'] = df_pivot['Metric'] + '_' + df_pivot['AtomID'].astype(str)
                
                # Create a single row DataFrame for this record
                df_single = df_pivot.pivot_table(index=[], columns='Metric', values='Value', aggfunc='first')
                df_single = df_single.reset_index(drop=True)

                # Add an identifier for this molecule based on the record
                ligand_name = record.split('|')[1].split(' ')[-1].split('-')[0]
                score = round(float(record.split('> <Gold.PLP.Fitness>')[1].split('> <Gold.PLP.PLP>')[0].strip()),3) 
                df_single['id'] = ligand_name
                df_single['score'] = score

                transformed_tables.append(df_single)

    # Concatenating all transformed DataFrames into a single DataFrame
    final_df = pd.concat(transformed_tables, ignore_index=True)
    final_df = final_df.fillna(0).reset_index(drop=True)
    return final_df

targets = pd.read_parquet('files/targets_information.parquet')
classes = ['decoys_dcm', 'decoys_znc', 'inactives']

for target in targets.name:
    active_df = padif_dataframe(f'docking_solutions/{target}_actives_solutions.sdf')
    active_df['activity'] = 1
    for cl in zip(classes, ['dcm', 'div', 'true']):
        decoys_df = padif_dataframe(f'docking_solutions/{target}_{cl[0]}_solutions.sdf')
        decoys_df['activity'] = 0
        padif = pd.concat([active_df, decoys_df])
        cols = ['id', 'score', 'activity']
        order = [col for col in padif.columns if col not in cols] + cols
        padif = padif[order]
        padif = padif.fillna(0.0)
        padif.to_csv(f'padif/{target}_{cl[1]}.parquet', index=False)

