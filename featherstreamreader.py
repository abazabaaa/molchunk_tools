

from pyarrow import Table

import pyarrow as pa




    
    
class InputStreamReader:
    def __init__(self, file_stream):
        self.file_stream = file_stream
        self._stream = None

    def batches(self):
        i = tries = 0
        while True:
            try:
                batch = self.__next_batch()
                i += 1
                yield i, batch
            except StopIteration:
                break

    def __next_batch(self):
        return self.stream.read_next_batch()


    @property
    def stream(self):
        if not self._stream:
#             read_options = pa.csv.ReadOptions(block_size=1048576000)
#             parse_options = pa.csv.ParseOptions(delimiter='\t')
#             convert_options = pa.csv.ConvertOptions(include_columns=include_columns)
            self._stream = pa.ipc.RecordBatchStreamReader(
                self.file_stream)

        
        return self._stream
