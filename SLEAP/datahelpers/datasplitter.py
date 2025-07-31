import os
import numpy as np
from datahelpers.data import Data

class InconsistentDataException(Exception):
    pass

def split_npz_by_size_streaming(input_path, output_dir, max_size_mb=100):

    max_size_bytes = max_size_mb * 1024 * 1024

    with np.load(input_path) as huge_file:
        array_names = huge_file.files
        if not array_names:
            raise FileExistsError(f"No array names within file data/{dataset_name}/{signal}/{data_file_name_within_signal}")
        
        # Get shape/dtype info from first array
        sample_array = huge_file[array_names[0]]
        total_items = len(sample_array)
        sample_item = sample_array[0]
        
        if hasattr(sample_item, 'nbytes'):
            bytes_per_item = sample_item.nbytes
        else:
            print("Ignorable Warning: Could not find size of sample item")
            bytes_per_item = 100

    items_per_signal.add(total_items)
    if len(items_per_signal) > 1:
        raise InconsistentDataException("Signal data does not have the same amount of items")
    
    items_per_file = max_size_bytes // bytes_per_item
    if items_per_file < 1:
        items_per_file = 1

    num_files = (total_items + items_per_file - 1) // items_per_file
    print(f"Splitting {total_items} items into {num_files} files (~{items_per_file} items per file)")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i in range(num_files):
        start = i * items_per_file
        end = min((i + 1) * items_per_file, total_items)

        chunk_data = {}
        with np.load(input_path) as huge_file:
            for column_name in array_names:
                chunk_data[column_name] = huge_file[column_name][start:end]

        output_path = os.path.join(output_dir, f"{base_name}_part{i+1}.npz")
        np.savez_compressed(output_path, **chunk_data)
        print(f"Saved {output_path} ({end-start} items)")

da = Data()
dataset_name = da.find_dataset()
items_per_signal = set()

for signal in da.get_all_signal_names():
    data_file_name_within_signal = os.listdir(f"data/{dataset_name}/{signal}")[1] #TODO: Breyta í 0, 1 útaf gitignore
    input_path = f"data/{dataset_name}/{signal}/{data_file_name_within_signal}"
    output_dir = f"data/{dataset_name}/{signal}/split_files"
    
    split_npz_by_size_streaming(input_path, output_dir, max_size_mb=100)