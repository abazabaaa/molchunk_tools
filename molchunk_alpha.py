%env OE_LICENSE=/data/openeye/oe_license.txt
from io import StringIO  
from openbabel import pybel
import tqdm
from mdtraj.utils.delay_import import import_
import os
import copy
import sys
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
import math
from rich import print
import pathlib
from rich.console import Console
console = Console()


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
def smiles_to_oemol(smiles,title='MOL'):
    """Create a OEMolBuilder from a smiles string.
    Parameters
    ----------
    smiles : str
        SMILES representation of desired molecule.
    Returns
    -------
    molecule : OEMol
        A normalized molecule with desired smiles string.
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))

    molecule = oechem.OEMol()
    if not oechem.OEParseSmiles(molecule, smiles):
        raise ValueError("The supplied SMILES '%s' could not be parsed." % smiles)

    molecule = normalize_molecule(molecule)

    # Set title.
    molecule.SetTitle(title)

    return molecule

@stopwatch
def generate_conformers(molecule, max_confs=800, strictStereo=True, ewindow=15.0, rms_threshold=1.0, strictTypes = True):
    """Generate conformations for the supplied molecule
    Parameters
    ----------
    molecule : OEMol
        Molecule for which to generate conformers
    max_confs : int, optional, default=800
        Max number of conformers to generate.  If None, use default OE Value.
    strictStereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
    strictTypes : bool, optional, default=True
        If True, requires that Omega have exact MMFF types for atoms in molecule; otherwise, allows the closest atom type of the same element to be used.
    Returns
    -------
    molcopy : OEMol
        A multi-conformer molecule with up to max_confs conformers.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    oeomega = import_("openeye.oeomega")
    if not oeomega.OEOmegaIsLicensed(): raise(ImportError("Need License for OEOmega!"))

    molcopy = oechem.OEMol(molecule)
    omega = oeomega.OEOmega()

    # These parameters were chosen to match http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(True)
    omega.SetCanonOrder(False)

    omega.SetSampleHydrogens(True)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetEnergyWindow(ewindow)
    omega.SetRMSThreshold(rms_threshold)  # Word to the wise: skipping this step can lead to significantly different charges!

    omega.SetStrictStereo(strictStereo)
    omega.SetStrictAtomTypes(strictTypes)

    omega.SetIncludeInput(False)  # don't include input
    if max_confs is not None:
        omega.SetMaxConfs(max_confs)

    status = omega(molcopy)  # generate conformation
    if not status:
        raise(RuntimeError("omega returned error code %d" % status))


    return molcopy

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
def molecule_to_mol2(molecule, tripos_mol2_filename=None, conformer=0, residue_name="MOL", standardize=True):
    """Convert OE molecule to tripos mol2 file.
    Parameters
    ----------
    molecule : openeye.oechem.OEGraphMol
        The molecule to be converted.
    tripos_mol2_filename : str, optional, default=None
        Output filename.  If None, will create a filename similar to
        name.tripos.mol2, where name is the name of the OE molecule.
    conformer : int, optional, default=0
        Save this frame
        If None, save all conformers
    residue_name : str, optional, default="MOL"
        OpenEye writes mol2 files with <0> as the residue / ligand name.
        This chokes many mol2 parsers, so we replace it with a string of
        your choosing.
    standardize: bool, optional, default=True
        Use a high-level writer, which will standardize the molecular properties.
        Set this to false if you wish to retain things such as atom names.
        In this case, a low-level writer will be used.
    Returns
    -------
    tripos_mol2_filename : str
        Filename of output tripos mol2 file
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for oechem!"))

    # Get molecule name.
    molecule_name = molecule.GetTitle()
    logger.debug(molecule_name)

    # Write molecule as Tripos mol2.
    if tripos_mol2_filename is None:
        tripos_mol2_filename = molecule_name + '.tripos.mol2'

    ofs = oechem.oemolostream(tripos_mol2_filename)
    ofs.SetFormat(oechem.OEFormat_MOL2H)
    for k, mol in enumerate(molecule.GetConfs()):
        if k == conformer or conformer is None:
            # Standardize will override molecular properties(atom names etc.)
            if standardize:
                oechem.OEWriteMolecule(ofs, mol)
            else:
                oechem.OEWriteMol2File(ofs, mol)

    ofs.close()

    # Replace <0> substructure names with valid text.
    infile = open(tripos_mol2_filename, 'r')
    lines = infile.readlines()
    infile.close()
    newlines = [line.replace('<0>', residue_name) for line in lines]
    outfile = open(tripos_mol2_filename, 'w')
    outfile.writelines(newlines)
    outfile.close()

    return molecule_name, tripos_mol2_filename

@stopwatch
def get_charges(molecule, max_confs=800, strictStereo=True,
                normalize=True, keep_confs=None, legacy=False):
    """Generate charges for an OpenEye OEMol molecule.
    Parameters
    ----------
    molecule : OEMol
        Molecule for which to generate conformers.
        Omega will be used to generate max_confs conformations.
    max_confs : int, optional, default=800
        Max number of conformers to generate
    strictStereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
        See https://docs.eyesopen.com/omega/usage.html
    normalize : bool, optional, default=True
        If True, normalize the molecule by checking aromaticity, adding
        explicit hydrogens, and renaming by IUPAC name.
    keep_confs : int, optional, default=None
        If None, apply the charges to the provided conformation and return
        this conformation, unless no conformation is present.
        Otherwise, return some or all of the generated
        conformations. If -1, all generated conformations are returned.
        Otherwise, keep_confs = N will return an OEMol with up to N
        generated conformations.  Multiple conformations are still used to
        *determine* the charges.
    legacy : bool, default=True
        If False, uses the new OpenEye charging engine.
        See https://docs.eyesopen.com/toolkits/python/quacpactk/OEProtonFunctions/OEAssignCharges.html#
    Returns
    -------
    charged_copy : OEMol
        A molecule with OpenEye's recommended AM1BCC charge selection scheme.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    # If there is no geometry, return at least one conformation.
    if molecule.GetConfs() == 0:
        keep_confs = 1

    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    oequacpac = import_("openeye.oequacpac")
    if not oequacpac.OEQuacPacIsLicensed(): raise(ImportError("Need License for oequacpac!"))

    if normalize:
        molecule = normalize_molecule(molecule)
    else:
        molecule = oechem.OEMol(molecule)

    charged_copy = generate_conformers(molecule, max_confs=max_confs, strictStereo=strictStereo)  # Generate up to max_confs conformers

    if not legacy:
        # try charge using AM1BCCELF10
        status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1BCCELF10Charges())
        # or fall back to OEAM1BCC
        if not status:
            # 2017.2.1 OEToolkits new charging function
            status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1BCCCharges())
            if not status:
                # Fall back
                status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1Charges())

                # Give up
                if not status:
                    raise(RuntimeError("OEAssignCharges failed."))
    else:
        # AM1BCCSym recommended by Chris Bayly to KAB+JDC, Oct. 20 2014.
        status = oequacpac.OEAssignPartialCharges(charged_copy, oequacpac.OECharges_AM1BCCSym)
        if not status: raise(RuntimeError("OEAssignPartialCharges returned error code %d" % status))

    #Determine conformations to return
    if keep_confs == None:
        #If returning original conformation
        original = molecule.GetCoords()
        #Delete conformers over 1
        for k, conf in enumerate( charged_copy.GetConfs() ):
            if k > 0:
                charged_copy.DeleteConf(conf)
        #Copy coordinates to single conformer
        charged_copy.SetCoords( original )
    elif keep_confs > 0:
        logger.debug("keep_confs was set to %s. Molecule positions will be reset." % keep_confs)

        #Otherwise if a number is provided, return this many confs if available
        for k, conf in enumerate( charged_copy.GetConfs() ):
            if k > keep_confs - 1:
                charged_copy.DeleteConf(conf)
    elif keep_confs == -1:
        #If we want all conformations, continue
        pass
    else:
        #Not a valid option to keep_confs
        raise(ValueError('Not a valid option to keep_confs in get_charges.'))

    return charged_copy

@stopwatch
def get_mol2_string_from_OEMol(molecule):
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    molecule_name = molecule.GetTitle()
    conformer=0
    standardize=True
#     print(molecule.GetConfs())



    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_MOL2H)
    ofs.openstring()
    for k, mol in enumerate(molecule.GetConfs()):
        if k == conformer or conformer is None:
            # Standardize will override molecular properties(atom names etc.)
            if standardize:
                oechem.OEWriteMolecule(ofs, mol)
            else:
                oechem.OEWriteMol2File(ofs, mol)

    molfile = ofs.GetString()
    return molfile

@stopwatch
def mol_chunk_from_smi(smileslist, nameslist, prefix, output_dir):
    
    names_list = []
    smiles_list = []
    mol2_block_am1bcc_list = []
    pdbqt_am1bcc_list = []
    pdbqt_gast_list = []
    pdb_am1bcc_list = []
    failed_smi = []
    failed_name = []

    for count,smile in enumerate(tqdm.tqdm_notebook(smileslist, total=len(smileslist), smoothing=0)):
        try:
            smiles = str(smileslist[count])
            name = nameslist[count]
            console.print(f'Converting {name} with smiles {smiles} to PDBQT', style="bold blue")

            mol = smiles_to_oemol(smiles,title=name)
    #         print(type(mol))
            mol2 = generate_conformers(mol, max_confs=800, strictStereo=True, ewindow=15.0, rms_threshold=1.0, strictTypes = True)
    #         print(type(mol2))
            mol3 = get_charges(mol2, max_confs=800, strictStereo=True,
            normalize=True, keep_confs=None, legacy=False)
    #         print(type(mol3))
            mol2_block = get_mol2_string_from_OEMol(mol3)

            mol2_block_string = mol2_block.decode("utf-8")
            mol2_pybel = pybel.readstring(format='mol2', string=mol2_block_string)

            pdbqt_am1bcc_list.append(mol2_pybel.write(format='pdbqt'))

            mol2_block_am1bcc_list.append(mol2_block)
            pdb_am1bcc_list.append(mol2_pybel.write(format='pdb'))

            mol2_pybel.calccharges(model='gasteiger')
            pdbqt_gast_list.append(mol2_pybel.write(format='pdbqt'))


            names_list.append(name)
            smiles_list.append(smiles)

            console.print(f'{name} completed', style="bold purple")
        except:
            console.print(f'failed to generate 3d structure, appending to failed list', style="bold red")
            failed_smi.append(str(smileslist[count]))
            failed_name.append(nameslist[count])
            pass
    
    
    
    molchunkpath = pathlib.Path(f'{output_dir}/{prefix}.molchunk')

    mol_chunk = pd.DataFrame(list(zip(names_list, smiles_list, mol2_block_am1bcc_list, pdb_am1bcc_list, pdbqt_am1bcc_list)), columns = ['names', 'smiles', 'mol_block_am1bcc', 'pdb_block_am1bcc', 'pdbqt_block_am1bcc'])
    feather.write_feather(mol_chunk, molchunkpath)

#     failed_molspath = pathlib.Path(f'{output_dir}/failed_{prefix}.molchunk')
    
#     mol_chunk_failed = pd.DataFrame(list(zip(failed_name, failed_smi)), columns = ['names', 'smiles'])
#     feather.write_feather(mol_chunk_failed, failed_molspath)

@stopwatch
def df_from_dude_smi_file(dude_smi_file, delimiter, name_col, smi_col):
    parse_options = pa.csv.ParseOptions(delimiter=delimiter)
    table = csv.read_csv(dude_smi_file, parse_options=parse_options)
    name_col = int(name_col)
    smi_col = int(smi_col)
    name_arr = table.column(name_col)
    smiles_arr = table.column(smi_col)

    data = [
        name_arr,
        smiles_arr
    ]

    table2 = pa.Table.from_arrays(data, names=['names', 'smiles'])
    df = pa.Table.to_pandas(table2)
    return df

dude_smi_file = '/home/schrogpu/Downloads/DRD3_chembl19_set.smi'
delimiter = ' '
df = df_from_dude_smi_file(dude_smi_file, delimiter, 1, 0)
smileslist = list((df['smiles'])[14090:])
nameslist = list((df['names'])[14090:])
print(len(smileslist))
print(len(nameslist))

prefix = 'mol_chunks_test'
output_dir = '/data/dopamine_3_results/mol_chunk_docking'
mol_chunk_from_smi(smileslist, nameslist, prefix, output_dir)