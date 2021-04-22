import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import glob
import pathlib
import ray
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem
from datetime import timedelta
from timeit import time
import numpy as np

import pyarrow.feather as feather

def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed

@ray.remote(memory=1000 * 1024 * 1024)
def enumerate_smiles_stereoisomers(job_count, record_batch_list):

    for x, record_batch in enumerate(record_batch_list):

        smiles = list(record_batch.column('smiles'))


        
        
        canonical_id = list(record_batch.column('canonical_id'))

        
    #     print(type(smiles))
    #     print(type(canonical_id))
    #     print(canonical_id[10])
    #     print(smiles[10])


         
        print(f'the number of smiles in the record batch is {len(smiles)}')




        
        canonical_id_list = []
        smiles_list = []

        for count,smi in enumerate(smiles):
            clean_smi = str(smi)
            mol = Chem.MolFromSmiles(clean_smi)
            canonical_id_prefix = str(canonical_id[count])
            opts = StereoEnumerationOptions(maxIsomers=20,rand=0xf00d)
            isomers = EnumerateStereoisomers(mol, options=opts)
            enum_smiles = sorted(Chem.MolToSmiles(y,isomericSmiles=True) for y in isomers)
            for count, smi in enumerate(enum_smiles):
                canonical_id_isomer_num = f'{canonical_id_prefix}_{count}'
                smiles_string = smi
                canonical_id_list.append(canonical_id_isomer_num)
                smiles_list.append(smiles_string)
            # except:
            #     print('molecule failed')
        outfilename = '/data/dopamine_3_results/enumerated_smiles'

        name = os.path.join(outfilename, 'enumerated_stereoisomers_'+str(x)+'_'+str(job_count))

        record_batch = namesdict_to_arrow_batch(canonical_id_list, smiles_list)
        
        df = record_batch.to_pandas()
        print(df.shape)

        feather.write_feather(df, f'{name}.feather')

        print(f'Job number {job_count} sub-batch {x} complete.')
        print(f'Job contained {len(smiles)} smiles strings')
        print(f'Job generated a total of {len(smiles_list)} smiles after enumeration')
        # time.sleep(10)
        


@stopwatch       
def namesdict_to_arrow_batch(canonical_id_list, smiles_list):
    smiles_list2 = [str(smiles_list[i]) for i in range(len(smiles_list))]
    # scores_list = np.array(scores_list, dtype=np.float16)
    name_list = pa.array(canonical_id_list, type=pa.string())
    smiles_list3 = pa.array(smiles_list2)
    data1 = [
        name_list,
        smiles_list3
    ]

    batch_from_arrays = pa.RecordBatch.from_arrays(data1, ['canonical_id', 'smiles'])
    return batch_from_arrays


dataset_path = pathlib.Path('/data/dopamine_3_results/test_combine_large/combined.parquet')

columns = ['smiles', \
           'canonical_id']

dataset = ds.dataset(dataset_path, format="parquet")
scan_tasks = [scan_task for scan_task in dataset.scan(columns=columns, batch_size=50000)]
enumerate_list = [(index, element) for index, element in enumerate(scan_tasks)]
ray.init(num_cpus=32)

futures = [enumerate_smiles_stereoisomers.remote(index, (list(element.execute()))) for index, element in enumerate_list]
results = [ray.get(f) for f in futures]
