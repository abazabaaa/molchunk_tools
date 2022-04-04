# %env OE_LICENSE=/data/openeye/oe_license.txt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PyMol
import os
import copy
import sys
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
import pandas as pd
from rdkit.Chem import PandasTools
from openbabel import pybel
from rich import print
import pathlib
from rich.console import Console
import subprocess
import re
import xml.etree.ElementTree as ET
from io import StringIO
from datetime import timedelta
from timeit import time
from mdtraj.utils.delay_import import import_
console = Console()
os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"

# cols = ['names', 'smiles', 'mol_block_am1bcc', 'pdb_block_am1bcc', 'pdbqt_block_am1bcc', 'pdbqt_gast_list']

# name_col = cols[0]
# smiles_col = cols[1]
# mol2_block_am1bcc = cols[2]
# pdb_block_am1bcc = cols[3]
# pdbqt_block_am1bcc = cols[4]
# pdbqt_gast_list = cols[5]

def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed

@stopwatch
def df_from_molchunk(molchunk_path):
    df = feather.read_feather(molchunk_path)
    # PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')
    return df

@stopwatch
def mols_to_pymol(df):
    v = PyMol.MolViewer()
    v.DeleteAll()
    mols = [Chem.MolFromMol2Block(m, removeHs=False) for m in df['mol_block_am1bcc']]
    names = list(df['names'])
    for count,m in enumerate(mols):
        molid = names[count]
        m.SetProp('_Name', molid)
        probe = Chem.Mol(m.ToBinary())
        v.ShowMol(probe, name=molid, showOnly=False)
        
@stopwatch        
def run_autodock_gpu(df, col_to_dock, autodock_gpu, lsmet, num_runs, working_dir, receptor_path):
    
    
    names = list(df['names'])
    pdbqt = df[col_to_dock]
    
    filenames = []
    names_to_dock = []
    for count,m in enumerate(pdbqt):
        pdbqt_block_string = m
        pdbqt_pybel = pybel.readstring(format='pdbqt', string=pdbqt_block_string)
#         print(pdbqt_pybel)
#         print(pdbqt_pybel.write(format='pdbqt'))
#         print(type(pdbqt_pybel))
        filename = f'{working_dir}/{names[count]}_{col_to_dock}.pdbqt'
        filenames.append(filename)
        names_to_dock.append(names[count])
        pdbqt_pybel.write(format='pdbqt', filename=filename)
        
    
    output_prefix_paths = []
    docked_names = []
    batch_list = f'{working_dir}/{col_to_dock}_batch.txt'    
    with open(batch_list, 'w') as f:
        f.write(f'{receptor_path}\n')
        for i, filepath in enumerate(filenames):
            f.write(f'{filepath}\n')
            output_prefix = f'{working_dir}/{names_to_dock[i]}_docked'
            f.write(f'{working_dir}/{names_to_dock[i]}_docked\n')
            output_prefix_paths.append(output_prefix)
            docked_names.append(names_to_dock[i])
    
    program_exe = '''\
    {autodock_gpu} \
    -filelist {batch_list} \
    -lsmet {lsmet} \
    -nrun {num_runs}
    '''
    
#         -autostop 0 \
#     -heuristics 0 \

    exe_cmd = program_exe.format(autodock_gpu=autodock_gpu, receptor_path=receptor_path, batch_list=batch_list, lsmet=lsmet, num_runs=num_runs)
    shell_script = '''\
    {exe_cmd}
    '''.format(exe_cmd=exe_cmd)
    print(shell_script)
    
    x = subprocess.Popen(shell_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_out, std_err = x.communicate()
    if std_err :
        print(std_err)

    else:
        x.wait()
        console.print('docking_complete')
        
    output_docking_poses_ext = ".dlg"
    scores_output_ext = ".xml"
    output_docking_pose_paths = [(f'{working_dir}/{name}_docked{output_docking_poses_ext}') for i,name in enumerate(docked_names)]
    output_docking_scores = [(f'{working_dir}/{name}_docked{scores_output_ext}') for i,name in enumerate(docked_names)]

        
    docked_df = pd.DataFrame(list(zip(names_to_dock, filenames, output_docking_pose_paths, output_docking_scores)), \
                             columns = ['names', 'input_pdbqt_path', 'output_docking_pose_paths', 'output_docking_scores'])
    
    
    try:
        out_df = pd.merge(df, docked_df, on="names")
        return out_df 
    except:
        print("merging df, failed")
        return None
    
    
        
# def extract_top_autodock_pose(target_dir, num_runs):
  
#     file_ext = ".dlg"
#     dlg_files_list = glob.glob(os.path.join(target_dir, "*"+file_ext))


#     for l in dlg_files_list:
        
#         print(f'opening dlg file {l}....')
#         file_name = pathlib.Path(l)


@stopwatch 
def extract_specific_pdbqt(df, num_runs, working_dir):
    
    dlg_paths = df['output_docking_pose_paths']
    target_runs = df['top_docking_run']
    names = df['names']
    
    mol2_blocks_docked = []
    names_lis = []
    
    for count,dlg in enumerate(dlg_paths):    
        file = open(dlg, "r")
#         print(file)
        lines = file.readlines()
#         print(len(lines))
#         target_dir = str(file_name.parent)
        target_run = target_runs[count]


        starting_line = 0
        ending_line = 0
#         print(f'Extracting Run:   {target_run} / {num_runs}')
        run_target = str(f'Run:   {target_run} / {num_runs}')
#         print(run_target)
        for line in lines:
            if line.startswith(run_target):
#                 print('found starting line of target')
                starting_line= lines.index(line)
#                 print(f'The startomg line is {starting_line}')
                break

        if target_run != num_runs:
            # print('the target was not 10')
            for line in lines[starting_line:]:
                end_run_target = f'Run:   {(int(target_run) + 1)} / {num_runs}'
                if line.startswith(str(end_run_target)):
#                     print('found ending line of target')
                    ending_line = lines.index(line)
#                     print(f'The ending line is {ending_line}')
                    break

#             print(f' the starting line is {starting_line} and the ending line is {ending_line}')
            pdbqt_content = lines[(starting_line + 4):(ending_line - 9)]
#             print(f' the pdbqt content is found on lines {pdbqt_content}')
            stripped_pdbqt_content = [line.rstrip('\n') for line in pdbqt_content]


        if target_run == num_runs:
            # print('the target was 10')
            for line in lines[starting_line:]:
                if line.startswith('    CLUSTERING HISTOGRAM'):
                    ending_line = lines.index(line)
                    break
#             print(f' the starting line is {starting_line} and the ending line is {ending_line}')
            pdbqt_content = lines[(starting_line + 4):(ending_line - 5)]
#             print(f' the pdbqt content is found on lines {pdbqt_content}')
            stripped_pdbqt_content = [line.rstrip('\n') for line in pdbqt_content]

        clean_pdbqt_content = []
#         print(f'there are {len(clean_pdbqt_content)} lines in the clean_pdbqt_content')
        for line in stripped_pdbqt_content:
            cleaned_line = line[max(line.find('D'), 8):]
            if not cleaned_line.startswith('USER'):
                clean_pdbqt_content.append(cleaned_line)
#         print(f'there are now {len(clean_pdbqt_content)} lines in the clean_pdbqt_content')

        pdbqt_block_content = []
#         print(f'there are {len(pdbqt_block_content)} lines in the pdbqt content')
        for line_item in clean_pdbqt_content:
            pdbqt_block_content.append("%s\n" % line_item)
#         print(f'there are now {len(clean_pdbqt_content)} lines in the pdbqt_block_content')
        
        pdbqt_name = f'{working_dir}/temp.pdbqt'
        with open(pdbqt_name, 'w') as f:
            for line_item in clean_pdbqt_content:
                f.write("%s\n" % line_item)
                
                
        stringpdbqt = ' '.join(map(str,  pdbqt_block_content))
        pdbqt_pybel = list(pybel.readfile("pdbqt", pdbqt_name))
#         print(list(pdbqt_pybel))
        mol2_blocks_docked.append(pdbqt_pybel[0].write(format='mol2'))
        names_lis.append(names[count])
        
    docked_out_df = pd.DataFrame(list(zip(names_lis, mol2_blocks_docked)), \
                                 columns = ['names', 'mol2_blocks_docked'])

    #     print(docked_out_df.shape)     
    output_df = pd.merge(df, docked_out_df, on="names")
    #     print(output_df.shape)   
    return output_df

@stopwatch 
def index_docking_output(df):
    xml_paths = list(df['output_docking_scores'])
    dlg_paths = list(df['output_docking_pose_paths'])
    names_lis = list(df['names'])

    docking_scores = []
    top_docking_run_lis = []
    names_to_extract = []
    for count,xml_path in enumerate(xml_paths):
        try:
            print(xml_path)
            print(names_lis[count])
            tree = ET.parse(xml_path)
            root = tree.getroot()
            docking_score = [x.get('lowest_binding_energy') for x in root.findall(".//*[@cluster_rank='1']")]
            top_docking_run = [y.get('run') for y in root.findall(".//*[@cluster_rank='1']")]
            print(docking_score[0])
            print(top_docking_run[0])
            docking_scores.append(docking_score[0])
            top_docking_run_lis.append(top_docking_run[0])
            names_to_extract.append(names_lis[count])

        except FileNotFoundError:
            print(f'There was an error with prep_docking_results')
            print(f'The target xml_path was: {xml_path}')
            print("Wrong file or file path") 
            pass


    extracted_scores = pd.DataFrame(list(zip(names_to_extract, top_docking_run_lis, docking_scores)), \
                             columns = ['names', 'top_docking_run', 'top_docking_score'])
    indexed_df = pd.merge(df, extracted_scores, on="names")
    return indexed_df


@stopwatch
def normalize_molecule(molecule):
    """
    Normalize a copy of the molecule by checking aromaticity, adding explicit hydrogens, and
    (if possible) renaming by IUPAC name.
    Parameters
    ----------
    molecule : OEMol
        the molecule to be normalized.
    Returns
    -------
    molcopy : OEMol
        A (copied) version of the normalized molecule
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed():
        raise(ImportError("Need License for OEChem!"))
    oeiupac = import_("openeye.oeiupac")
    has_iupac = oeiupac.OEIUPACIsLicensed()

    molcopy = oechem.OEMol(molecule)

    # Assign aromaticity.
    oechem.OEAssignAromaticFlags(molcopy, oechem.OEAroModelOpenEye)

    # Add hydrogens.
    oechem.OEAddExplicitHydrogens(molcopy)

    # Set title to IUPAC name.
    if has_iupac:
        name = oeiupac.OECreateIUPACName(molcopy)
        molcopy.SetTitle(name)

    # Check for any missing atom names, if found reassign all of them.
    if any([atom.GetName() == '' for atom in molcopy.GetAtoms()]):
        oechem.OETriposAtomNames(molcopy)

    return molcopy


@stopwatch        
def oe_mol2_to_mol2_block(mol2_block):
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_MOL2)
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SDF)
    oms.openstring()
    
    mol = oechem.OEGraphMol()

    while oechem.OEReadMolecule(ifs, mol):
        oechem.OEWriteMolecule(ofs, mol)
    
    mols = []
    mol = oechem.OEMol()
    if ifs.open(out_sdf_path):
        for mol in ifs.GetOEGraphMols():
            mols.append(oechem.OEMol(mol))

    else:
        oechem.OEThrow.Fatal(f"Unable to open {out_sdf_path}")
#     print(type(mols[0]))
    molecule = mols[0]
    molecule = normalize_molecule(molecule)
    
    oechem.OEWriteMolecule(oms, mol)

    molfile = oms.GetString()
    print("MOL string\n", molfile.decode('UTF-8'))
    return molecule

# @stopwatch
# def docked_pdbqt_to_pymol(df):
#     v = PyMol.MolViewer()
#     v.DeleteAll()
#     pdbqt_blocks = list(df['docked_pdbqt_block'])
#     names = list(df['names'])
#     for count,pdbqt in enumerate(pdbqt_blocks):
        
#         pdbqt_block_string = pdbqt.decode("utf-8")
#         pdbqt_pybel = pybel.readstring(format='pdbqt', string=pdbqt_block_string)
#         molblock = StingIO(pdbqt_pybel.write(format='mol2'))
#         mols = Chem.MolFromMol2Block(molblock, removeHs=False)
# #         names = list(df['names'])
# #         for count,m in enumerate(mols):
# #             molid = names[count]
# #             m.SetProp('_Name', molid)
# #             probe = Chem.Mol(m.ToBinary())
# #             v.ShowMol(probe, name=molid, showOnly=False)
# @stopwatch
# def mol2_string_IO_san(mol2_block_string):
    
#     oechem = import_("openeye.oechem")
#     if not oechem.OEChemIsLicensed():
#         raise(ImportError("Need License for OEChem!"))
#     mol2_block = mol2_block_string.encode('UTF-8')


#     ims = oechem.oemolistream()
#     ims.SetFormat(oechem.OEFormat_MOL2)
#     ims.openstring(mol2_block)

#     mols = []
#     mol = oechem.OEMol()
#     for mol in ims.GetOEMols():
#         mols.append(oechem.OEMol(mol))

#     oms = oechem.oemolostream()
#     oms.SetFormat(oechem.OEFormat_PDB)
#     oms.openstring()

#     for mol in mols:

#         mol2 = normalize_molecule(mol)

#         oechem.OEWriteMolecule(oms, mol2)

#     molfile = oms.GetString()
# #     print("MOL string\n", molfile.decode('UTF-8'))
#     return molfile

def mol2_string_IO_san(mol2_block_string):
    
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    mol2_block = mol2_block_string.encode('UTF-8')


    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_MOL2)
    ims.openstring(mol2_block)

    mols = []
    mol = oechem.OEMol()
    for mol in ims.GetOEMols():
        mols.append(oechem.OEMol(mol))

    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_PDB)
    oms.openstring()

    for mol in mols:

        mol2 = normalize_molecule(mol)

        oechem.OEWriteMolecule(oms, mol2)

    molfile = oms.GetString()
#     print("MOL string\n", molfile.decode('UTF-8'))
    return molfile

@stopwatch
def show_docked(df):
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    # def show_docked(df):

    mol2_blocks_docked = list(df['mol2_blocks_docked'])
    smiles_template = list(df['smiles'])
    names = list(df['names'])
    v = PyMol.MolViewer()
    v.DeleteAll()
    for count,molblock in enumerate(mol2_blocks_docked):

        molout = mol2_string_IO_san(molblock)
        mol = Chem.MolFromPDBBlock(molout)
        template = Chem.MolFromSmiles(smiles_template[count])
        new_mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    #     mol2_blocks_template = Chem.MolFromMol2Block(molblock)
        print(type(new_mol))

        molid = names[count]
        print(molid)
        mol.SetProp('_Name', molid)

        probe = Chem.Mol(new_mol.ToBinary())
        v.ShowMol(probe, name=molid, showOnly=False)

def remove_duplicates(df):

    if df[df.duplicated(['names'])].shape[0] == 0:
        print('No duplicate names detected')

    else:
        print('duplicate name entries detected')
        duplicateRowsDF = df[df.duplicated(['names'])]
        print("Duplicate Rows based on a single column are:", duplicateRowsDF, sep='\n')

        indices = duplicateRowsDF.index
        names = duplicateRowsDF['names']

        dupenum = 2
        for count,indexid in enumerate(indices):
            print(df.loc[indexid,'names'])
            new_value = f'{names[indexid]}_{str(dupenum)}'
            print(new_value)
            df.at[indexid,'names'] = new_value
    return df


molchunk_path = '/data/dopamine_3_results/mol_chunk_docking/mol_chunks_test_2001_5000.molchunk'

autodock_gpu = '/home/schrogpu/ADFRsuite-1.0/AutoDock-GPU/bin/autodock_gpu_128wi'
receptor_path = '/home/schrogpu/ADFRsuite-1.0/pocket2_fixer_moreatoms/rigidReceptor.maps.fld'
lsmet = 'sw'
num_runs = 50
col_to_dock = 'pdbqt_block_am1bcc'
# col_to_dock = 'pdbqt_gast_list'
working_dir = '/data/dopamine_3_results/mol_chunk_docking/test_2001_5000'
df = df_from_molchunk(molchunk_path)
df = remove_duplicates(df)
df = remove_duplicates(df)
df = remove_duplicates(df)
# mols_to_pymol(df)
outdf = run_autodock_gpu(df, col_to_dock, autodock_gpu, lsmet, num_runs, working_dir, receptor_path)
indexed_df = index_docking_output(outdf)
final_df = extract_specific_pdbqt(indexed_df, num_runs, working_dir)
# show_docked(final_df)
# del final_df['ROMol']
feather.write_feather(final_df, '/data/dopamine_3_results/mol_chunk_docking/mol_chunks_test_2001_5000_out.molchunk')