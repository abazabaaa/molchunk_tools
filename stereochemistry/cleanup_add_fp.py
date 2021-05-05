import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
from pyarrow.parquet import ParquetWriter

import rdkit
print(rdkit.__version__)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


import datamol as dm


from rich.console import Console
import subprocess





import numpy as np
import numpy.ma as ma
from scipy import sparse


import tqdm

import math
import time
from functools import reduce

import pathlib

import operator

from datetime import timedelta
from timeit import time
dm.disable_rdkit_log()
#autocomplete wasn't working for some reason. This fixes it. 
# %config Completer.use_jedi = False



console = Console()

def _preprocess(i, row):
#     print('hello')
    try:
        mol = dm.to_mol(str(row[smiles_column]), ordered=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)
        
        fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
        pars = { "radius": 2,
                         "nBits": 8192,
                         "invariants": [],
                         "fromAtoms": [],
                         "useChirality": True,
                         "useBondTypes": True,
                         "useFeatures": False,
                }
        pars2 = { "radius": 2,
                         "nBits": 8192,
                         "invariants": [],
                         "fromAtoms": [],
                         "useChirality": False,
                         "useBondTypes": True,
                         "useFeatures": False,
                }
                
        fp = fingerprint_function(mol, **pars)
        fp1 = fingerprint_function(mol, **pars2)

        row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
        row["selfies"] = dm.to_selfies(mol)
        row["inchi"] = dm.to_inchi(mol)
        row["inchikey"] = dm.to_inchikey(mol)
        row["onbits_fp_chiral"] =list(fp.GetOnBits())
        row["onbits_fp_achiral"] =list(fp1.GetOnBits())
        
        return row

    except ValueError:
        row["standard_smiles"] = 'dropped'
        row["selfies"] = 'dropped'
        row["inchi"] = 'dropped'
        row["inchikey"] = 'dropped'
        row["enumerated_smiles"] = list('dropped')
        return row

def fingerprint_matrix_from_df(df):
    smiles = list(df['standard_smiles'])
    onbits_fp = list(df['onbits_fp_achiral'])
    zincid = list(df['canonical_id'])
    # pars = { "radius": 2,
    #          "nBits": 8192,
    #          "invariants": [],
    #          "fromAtoms": [],
    #          "useChirality": False,
    #          "useBondTypes": True,
    #          "useFeatures": False,
    #          }

    # fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
    print(f'the number of smiles in the record batch is {len(smiles)}')
    count_ligs = len(smiles)



    # scores_list = []
    name_list =[]
    # smiles_list = []
    row_idx = list()
    col_idx = list()
    num_on_bits = []
    for count,m in enumerate(smiles):
    #     m_in = str(m)
    #     mol = Chem.MolFromSmiles(m_in)
    #     fp = fingerprint_function(mol, **pars)
    #     score = str(scores[count])
        zincid_name = str(zincid[count])
        onbits = list(onbits_fp[count])
    #     print(onbits)

    #     print(type(onbits))
        col_idx+=onbits
        row_idx += [count]*len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)
    #     scores_list.append(score)
        name_list.append(zincid_name)


        # except:
        #     print('molecule failed')

    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    fp_mat = fingerprint_matrix
    print('Fingerprint matrix shape:', fp_mat.shape)
    print('\n')
    print('Indices:', fp_mat.indices)
    print('Indices shape:', fp_mat.indices.shape)
    print('\n')
    print('Index pointer:', fp_mat.indptr)
    print('Index pointer shape:', fp_mat.indptr.shape)
    print('\n')
    print('Actual data (these are all just "ON" bits!):', fp_mat.data)
    print('Actual data shape:', fp_mat.data.shape)
    return fp_mat

# @stopwatch
def fast_dice(X, Y=None):
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X).astype(bool).astype(int)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y).astype(bool).astype(int)
            
    intersect = X.dot(Y.T)
    #cardinality = X.sum(1).A
    cardinality_X = X.getnnz(1)[:,None] #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    cardinality_Y = Y.getnnz(1) #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    return (1-(2*intersect) / (cardinality_X+cardinality_Y.T)).A

# @stopwatch
def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A

# @stopwatch
def _show_nearest_neighbor(i, row):

    """Use the output matrix from similarity search to find nearest neighbors from the
    reference matrix. out_array must be a globally defined variable for this to work.

    """
    a = out_array[i]




    minval = np.min(ma.masked_where(a==0, a)) 


#     maxval = np.max(ma.masked_where(a==0, a)) 



    minvalpos = np.argmin(ma.masked_where(a==0, a))  
    avgval = np.mean(ma.masked_where(a==0, a))


#     maxvalpos = np.argmax(ma.masked_where(a==0, a))  

    
    smiles_nn = smiles[minvalpos]
    name_nn = name[minvalpos]



    row["nearest_neighbor_smiles"] = smiles_nn 
    row["nearest_neighbor_name"] = name_nn
    row["nearest_neighbor_distance"] = minval
    row["average_sim_score"] = avgval
    return row

# @stopwatch
def ingest_chembl_smi(smi_path, smiles_column, canonical_id_column, activity_column):
    
    """Convert an smi file with a smiles column to a molchunk. It is assumed that
        the SMI has been cleaned (no header, and other columns have been removed).
        
    Args:
        smi_path: path to the smi file.
        smiles_column: column where the SMILES are located: f0 = col 1 f1 = col 2 .. etc
        canonical_id_column: name/id for molecule: f0 = col 1 f1 = col 2 .. etc
        activity column: column where bioactivity is listed (ki, ec50, etc): f0 = col 1 f1 = col 2 .. etc

    """

    
    # Next we will the multithreaded read options that pyarrow allows for.

    opts = pa.csv.ReadOptions(use_threads=True, autogenerate_column_names=True)

    # Then we tell pyarrow that the columns in our csv file are seperated by ';'
    # If they were tab seperated we would use '\t' and if it was comma we would use 
    # ','
    parse_options= pa.csv.ParseOptions(delimiter=' ')

    # Now we read the CSV into a pyarrow table. This is a columular dataset. More
    # on this later. Note how we specified the options above.

    table = pa.csv.read_csv(smi_path, opts, parse_options)


    # Now we will use a function that converts the pyarrow table into a pandas 
    # dataframe. We could have done this without arrow, but again -- there are 
    # very powerful tools that arrow will grant us.

    df_new = table.to_pandas()
 
    smiles_column = 'f0'
    
    # run initial mapper on smiles column to generate basic information and fingerprint on bits
    df_clean_mapped = dm.parallelized(_preprocess, list(df_new.iterrows()), arg_type='args', progress=True)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    
    #rename columns
    df_clean_mapped['smiles'] = df_clean_mapped[smiles_column]
    df_clean_mapped['canonical_id'] = df_clean_mapped[canonical_id_column]
    df_clean_mapped['ki'] = df_clean_mapped[activity_column]
    
    #delete old columns
    del df_clean_mapped['f2']
    del df_clean_mapped['f1']
    del df_clean_mapped['f0']
    
    #remove duplicated standard SMILES and reindex
    duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
#     print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
    df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
    df = df_clean_mapped.reset_index(drop=True)
    
    return df
# from rdkit import Chem
# import warnings

# def has_all_chiral_defined_rd(smiles, i):
#     try:
#         undefined_atoms = []
#         unspec_chiral = False
#         mol = Chem.MolFromSmiles(smiles)
#         mol = Chem.AddHs(mol)
#         chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
#         for center in chiral_centers:
#             atom_id = center[0]
#             if center[-1] == '?':
#                 unspec_chiral = True
#                 undefined_atoms.append((atom_id, mol.GetAtomWithIdx(atom_id).GetSmarts()))
#         if unspec_chiral:
#             print(undefined_atoms)
#             print(enum_smiles_lis[i])
#             console.print('False')
#             return False, undefined_atoms
#         else:
#             return True
#     except:
#         return False
    
# def mols_to_pymol(mols, names):
#     v = PyMol.MolViewer()
#     v.DeleteAll()
#     mols_3d = [dm.conformers.generate(m, n_confs=1) for m in mols]
# #     names = list(df['names'])
#     for count,m in enumerate(mols_3d):
#         molid = names[count]
#         m.SetProp('_Name', molid)
        
#         probe = Chem.Mol(m.ToBinary())
#         v.ShowMol(probe, name=molid, showOnly=False)


# Ingest a preprocessed smi file and generate a fingerprint matrix
smi_path = '/cbica/home/grahamth/molchunktools/molchunk_tools/test/d3_chembl.smi'
smiles_column = 'f0'
canonical_id_column = 'f1'
activity_column = 'f2'
d3_df = ingest_chembl_smi(smi_path, smiles_column, canonical_id_column, activity_column)

fingerprint_matrix_chembld3 = fingerprint_matrix_from_df(d3_df)


#define smiles and names for compounds in the matrix to be compared with
#this will be the key system for returning the nearest neighbor
smiles = list(d3_df['standard_smiles'])
name = list(d3_df['canonical_id'])

dataset_dir = '/cbica/home/grahamth/er_molchunk_dir'
dataset = ds.dataset(dataset_dir, format="feather")

output_dir = '/cbica/home/grahamth/d3fpsim'

# Create a list of fragments that are not memory loaded
fragments = [file for file in dataset.get_fragments()]


for count, element in enumerate(fragments):
    #cast the fragment as a pandas df
    df = element.to_table().to_pandas()
    #reset the index
    df = df.reset_index(drop=True)



    columns_to_keep = ['enumerated_smiles', 'CatalogID', 'ID_Index']
    df2 = df[columns_to_keep]
    df3 = df2.explode('enumerated_smiles')
    df5 = df3.reset_index(drop=True)
    df5 = df5.set_index('ID_Index')
    df5.index = df5.index + '_' + df5.groupby(level=0).cumcount().add(1).astype(str).replace('0','')
    df5 = df5.reset_index()
    df5.columns = ['canonical_id', 'enumerated_smiles', 'CatalogID']

    # columns_to_keep = ['enumerated_smiles', 'CatalogID', 'ID_Index']
    # df2 = df[columns_to_keep]
    # df3 = df2.explode('enumerated_smiles')
    # df5 = df3.reset_index(drop=True)
    smiles_column = 'enumerated_smiles'

    # run initial mapper on smiles column to generate basic information and fingerprint on bits
    df_clean_mapped = dm.parallelized(_preprocess, list(df5.iterrows()), arg_type='args', progress=True, n_jobs=4)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)

    df_dropped = df_clean_mapped[df_clean_mapped.standard_smiles == 'dropped']
    print(f' The number of dropped entries is: {len(df_dropped)})')
    feather.write_feather(df_dropped, f'{output_dir}/er_d3sim_dropped_{count}.molchunk')

    df_clean_mapped = df_clean_mapped[df_clean_mapped.standard_smiles != 'dropped']
    print(f' The number of successful entries is: {len(df_clean_mapped)})')



    # del df_clean_mapped['combined_smiles']


    #remove duplicated standard SMILES and reindex
    duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
    #     print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
    df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
    df6 = df_clean_mapped.reset_index(drop=True)


        #make the fp matrix
    fingerprint_matrix_in = fingerprint_matrix_from_df(df6)
    
    #make the jaccard matrix
    out_array = fast_jaccard(fingerprint_matrix_in, fingerprint_matrix_chembld3)

    #now write the nearest neighbor name and smiles to the df
    smiles_column = 'standard_smiles'
    df_add_nn = dm.parallelized(_show_nearest_neighbor, list(df6.iterrows()), arg_type='args', progress=True, n_jobs=4)
    df_add_nn = pd.DataFrame(df_add_nn)
    
    #write the mochunk to disk
    feather.write_feather(df_add_nn, f'{output_dir}/er_d3sim_{count}.molchunk')


# df = pa.feather.read_feather('/data/mol_chunk_tests_cluster/test_2.molchunk')
# # df['combined_smiles'] = df[['standard_smiles', 'enumerated_smiles']].values.tolist()
# columns_to_keep = ['enumerated_smiles', 'CatalogID', 'ID_Index']
# df2 = df[columns_to_keep]
# df3 = df2.explode('enumerated_smiles')
# df5 = df3.reset_index(drop=True)
# smiles_column = 'enumerated_smiles'

# # run initial mapper on smiles column to generate basic information and fingerprint on bits
# df_clean_mapped = dm.parallelized(_preprocess, list(df5.iterrows()), arg_type='args', progress=True)
# df_clean_mapped = pd.DataFrame(df_clean_mapped)



# # del df_clean_mapped['combined_smiles']


# #remove duplicated standard SMILES and reindex
# duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
# #     print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
# df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
# df6 = df_clean_mapped.reset_index(drop=True)
