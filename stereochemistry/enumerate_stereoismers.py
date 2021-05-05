import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import numpy.ma as ma
import tqdm
import pyarrow.parquet as pq
import numpy as np

import pandas as pd
import math
import time
from functools import reduce
import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import pathlib

import datamol as dm
import operator
import sys
from datetime import timedelta
from timeit import time
dm.disable_rdkit_log()

dataset_dir = sys.argv[1]
output_dir = sys.argv[2]

def _preprocess(i, row):
#     print('hello')
    try:
        mol = dm.to_mol(str(row[smiles_column]), ordered=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)
        opts = StereoEnumerationOptions(unique=True,maxIsomers=20,rand=0xf00d)
        isomers = EnumerateStereoisomers(mol, options=opts)
        enum_smiles = sorted(Chem.MolToSmiles(y,isomericSmiles=True) for y in isomers)
        
        smiles_list = []
        for count, smi in enumerate(enum_smiles):
            smiles_string = smi

            smiles_list.append(smiles_string)
        # fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
        # pars = { "radius": 2,
        #                  "nBits": 8192,
        #                  "invariants": [],
        #                  "fromAtoms": [],
        #                  "useChirality": False,
        #                  "useBondTypes": True,
        #                  "useFeatures": False,
        #         }
        # fp = fingerprint_function(mol, **pars)

        row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
        row["selfies"] = dm.to_selfies(mol)
        row["inchi"] = dm.to_inchi(mol)
        row["inchikey"] = dm.to_inchikey(mol)
        row["enumerated_smiles"] = smiles_list
        # row["onbits_fp"] =list(fp.GetOnBits())
        
        return row

    except ValueError:
        row["standard_smiles"] = 'dropped'
        row["selfies"] = 'dropped'
        row["inchi"] = 'dropped'
        row["inchikey"] = 'dropped'
        row["enumerated_smiles"] = list('dropped')
        return row

# Load the dataset from parquet one by one
dataset = ds.dataset(dataset_dir, format="parquet")

# Create a list of fragments that are not memory loaded
fragments = [file for file in dataset.get_fragments()]


for count, element in enumerate(fragments):
    #cast the fragment as a pandas df
    df_docked = element.to_table().to_pandas()
    #reset the index
    df_docked = df_docked.reset_index(drop=True)
       
    #now write the nearest neighbor name and smiles to the df
    smiles_column = 'Smile'
    df_add_nn = dm.parallelized(_preprocess, list(df_docked.iterrows()), arg_type='args', progress=True, n_jobs=54)
    df_add_nn = pd.DataFrame(df_add_nn)
    
    #write the mochunk to disk
    feather.write_feather(df_add_nn, f'{output_dir}/er_enumisomers_{count}.molchunk')
