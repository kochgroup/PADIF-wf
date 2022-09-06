"""
Workflow to extract PADIF, PADIF+, PROLIF, ECIF from GOLD docking results for train machine learning models

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

### Avoid unuseful warnings 
warnings.filterwarnings("ignore")

### Name for create a new folder / Its the name of the last folder 
os.chdir(argv[1])
targetName = str(argv[1]).split("/")[-2]
print(targetName)