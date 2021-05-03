import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools
import mols2grid

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PyMol
from IPython.display import SVG
import rdkit
print(rdkit.__version__)
import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rich.console import Console
import subprocess

console = Console()

def _preprocess(i, row):
#     print('hello')
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
    fp = fingerprint_function(mol, **pars)

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    row["selfies"] = dm.to_selfies(mol)
    row["inchi"] = dm.to_inchi(mol)
    row["inchikey"] = dm.to_inchikey(mol)
    row["onbits_fp"] =list(fp.GetOnBits())
    
    return row

from rdkit import Chem
import warnings

def has_all_chiral_defined_rd(smiles, i):
    try:
        undefined_atoms = []
        unspec_chiral = False
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for center in chiral_centers:
            atom_id = center[0]
            if center[-1] == '?':
                unspec_chiral = True
                undefined_atoms.append((atom_id, mol.GetAtomWithIdx(atom_id).GetSmarts()))
        if unspec_chiral:
            print(undefined_atoms)
            print(enum_smiles_lis[i])
            console.print('False')
            return False, undefined_atoms
        else:
            return True
    except:
        return False
    
def mols_to_pymol(mols, names):
    v = PyMol.MolViewer()
    v.DeleteAll()
    mols_3d = [dm.conformers.generate(m, n_confs=1) for m in mols]
#     names = list(df['names'])
    for count,m in enumerate(mols_3d):
        molid = names[count]
        m.SetProp('_Name', molid)
        
        probe = Chem.Mol(m.ToBinary())
        v.ShowMol(probe, name=molid, showOnly=False)
        
df = pa.feather.read_feather('/data/mol_chunk_tests_cluster/test_2.molchunk')
# df['combined_smiles'] = df[['standard_smiles', 'enumerated_smiles']].values.tolist()
columns_to_keep = ['enumerated_smiles', 'CatalogID', 'ID_Index']
df2 = df[columns_to_keep]
df3 = df2.explode('enumerated_smiles')
df5 = df3.reset_index(drop=True)
smiles_column = 'enumerated_smiles'

# run initial mapper on smiles column to generate basic information and fingerprint on bits
df_clean_mapped = dm.parallelized(_preprocess, list(df5.iterrows()), arg_type='args', progress=True)
df_clean_mapped = pd.DataFrame(df_clean_mapped)



# del df_clean_mapped['combined_smiles']


#remove duplicated standard SMILES and reindex
duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
#     print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
df6 = df_clean_mapped.reset_index(drop=True)

limit = 13000
results_list = []
names_lis = []
smiles_lis = []
for i, row in enumerate(df6.itertuples(), 1):
    if i>= limit: break
#     if (has_all_chiral_defined_rd(row.standard_smiles))[0] == False:
    
#     enum_smiles_results = [has_all_chiral_defined_rd(_) for _ in row.enumerated_smiles]
    results = has_all_chiral_defined_rd(row.standard_smiles, i)

    
    if results != True:
        results_list.append(results)
        names_lis.append(row.CatalogID)
        smiles_lis.append(row.standard_smiles)
        
        console.print(i, row.standard_smiles)
        console.print(i, row.CatalogID)
        
        
mols = [dm.to_mol(_) for _ in smiles_lis]
mols_to_pymol(mols, names_lis)
legends = [_ for _ in names_lis]
dm.viz.to_image(mols, legends=legends)
