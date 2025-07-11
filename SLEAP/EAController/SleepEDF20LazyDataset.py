# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from bisect import bisect_right

# import time

# class SleepEDF20LazyDataset(Dataset):
#     def __init__(self, x_files, y_files, data_dir, stage_map, max_open_files=153):
#         self.x_files = x_files
#         self.y_files = y_files
#         self.data_dir = data_dir
#         self.stage_map = stage_map
#         self.max_open_files = max_open_files  # Max open file descriptors
#         self.total_len = 0
#         self.total_time = 0
#         self.cache_misses = 0
        
#         # Cache management for memmap arrays
#         self.cache = {}         # Maps filename to memmap array
#         self.usage_order = []    # LRU tracking for open files
        
#         self._index_files()

#     def _index_files(self):
#         self.index_map = []  # Stores tuples of (x_file, y_file)
#         self.lengths = []
        
#         for x_file, y_file in zip(self.x_files, self.y_files):
#             x_path = os.path.join(self.data_dir, x_file)
            
#             # Read array header to get length without loading full data
#             with open(x_path, 'rb') as f:
#                 shape, _, _ = np.lib.format.read_array_header_1_0(f)
#                 n_samples = shape[0]
            
#             self.index_map.append((x_file, y_file))
#             self.lengths.append(n_samples)
#             self.total_len += n_samples
        
#         # Cumulative index ranges for efficient lookup
#         self.cumulative_lengths = np.cumsum([0] + self.lengths)

#     def __len__(self):
#         return self.total_len

#     def _load_file(self, file_path):
#         """Load file with memory mapping and manage cache using LRU policy"""
#         if file_path in self.cache:
#             if file_path in self.usage_order:
#                 self.usage_order.remove(file_path)
#             self.usage_order.append(file_path)
#             return self.cache[file_path]
#         else:
#             self.cache_misses += 1
        
#         # Load new file with memory mapping
#         memmap_array = np.load(file_path, mmap_mode='r')
#         self.cache[file_path] = memmap_array
#         self.usage_order.append(file_path)
        
#         # Evict least recently used files if over limit
#         while len(self.cache) > self.max_open_files and self.usage_order:
#             file_to_remove = self.usage_order.pop(0)
#             del self.cache[file_to_remove]
        
#         return memmap_array

#     def __getitem__(self, idx):
#         s = time.time()
#         # Find which file pair contains this index
#         file_idx = bisect_right(self.cumulative_lengths, idx) - 1
#         x_file, y_file = self.index_map[file_idx]
#         local_idx = idx - self.cumulative_lengths[file_idx]
        
#         # Get full paths
#         x_path = os.path.join(self.data_dir, x_file)
#         y_path = os.path.join(self.data_dir, y_file)
        
#         # Load using memory mapping with cache management
#         x_memmap = self._load_file(x_path)
#         y_memmap = self._load_file(y_path)
        
#         # Access data directly from memory-mapped arrays
#         x_data = x_memmap[local_idx].astype(np.float32)
#         y_data = y_memmap[local_idx]
        
#         # Apply transformations
#         x_data = x_data.transpose()  # Maintain original transpose
#         y_data = self.stage_map.get(y_data, 0)  # Map sleep stage

#         self.total_time += time.time() - s
        
#         return torch.tensor(x_data), torch.tensor(y_data)


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from bisect import bisect_right

from Globals import DataManager

class SleepEDF20LazyDataset(Dataset):
    def __init__(self, files, data_dir, stage_map):
        self.files = files
        self.data_dir = data_dir
        self.stage_map = stage_map
        self.max_mb = DataManager.MAX_MEMORY
        self.total_len = 0
        
        # Cache management
        self.cache = {}
        self.usage_order = []
        self.current_memory = 0  # bytes
        
        self._index_files()

    def _index_files(self):
        self.index_map = []
        self.lengths = []
        
        for file in self.files:
            path = os.path.join(self.data_dir, file)
            with np.load(path) as data:
                n_samples = len(data['x'])
            
            self.index_map.append(file)
            self.lengths.append(n_samples)
            self.total_len += n_samples
        
        # Cumulative index ranges for efficient lookup
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Find which file contains this index using bisect_right
        file_idx = bisect_right(self.cumulative_lengths, idx) - 1
        file = self.index_map[file_idx]
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Update cache if needed
        if file not in self.cache:
            path = os.path.join(self.data_dir, file)
            with np.load(path) as data:
                x = data['x']
                y = data['y']
            
            # Add to cache
            self.cache[file] = {'x': x, 'y': y}
            self.current_memory += x.nbytes + y.nbytes
            self._evict_to_fit()
        
        # Update usage order (move to end)
        if file in self.usage_order:
            self.usage_order.remove(file)
        self.usage_order.append(file)
        
        # Get data from cache
        data_dict = self.cache[file]
        x: torch.Tensor = data_dict['x'][local_idx].astype(np.float32)
        x = x.transpose()
        y = data_dict['y'][local_idx]
        y = self.stage_map.get(y, 0)
        
        return torch.tensor(x), torch.tensor(y)

    def _evict_to_fit(self):
        max_bytes = self.max_mb * 1024 * 1024
        while self.current_memory > max_bytes and self.usage_order:
            # Remove least recently used file
            file_to_remove = self.usage_order.pop(0)
            data_dict = self.cache.pop(file_to_remove)
            freed = data_dict['x'].nbytes + data_dict['y'].nbytes
            self.current_memory -= freed