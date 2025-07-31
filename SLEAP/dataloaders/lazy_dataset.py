import os
import numpy as np
import torch
from torch.utils.data import Dataset
from bisect import bisect_right

from Globals import DataManager

class LazyDataset(Dataset):
    def __init__(self, files, data_dir, use_stage_map, max_memory=DataManager.MAX_MEMORY):
        self.files = files
        self.data_dir = data_dir
        self.use_stage_map = use_stage_map
        self.max_mb = max_memory
        self.total_len = 0
        
        # Cache management
        self.cache = {}
        self.usage_order = []
        self.current_memory = 0  # bytes

        self.i = 0
        self.fit = False
        
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
        if self.fit:
            data_dict = self.cache[file]
            x: torch.Tensor = data_dict['x'][local_idx].astype(np.float32)
            x = x.transpose()
            y = data_dict['y'][local_idx]
            y = self.use_stage_map.get(y, 0) if self.use_stage_map else y
            
            return torch.tensor(x), torch.tensor(y)
        
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
            if len(self.cache) == len(self.files):
                self.fit = True

        
        # Update usage order (move to end)
        if file in self.usage_order:
            self.usage_order.remove(file)
        self.usage_order.append(file)
        
        # Get data from cache
        data_dict = self.cache[file]
        x: torch.Tensor = data_dict['x'][local_idx].astype(np.float32)
        x = x.transpose()
        y = data_dict['y'][local_idx]
        y = self.use_stage_map.get(y, 0) if self.use_stage_map else y
        
        return torch.tensor(x), torch.tensor(y)

    def _evict_to_fit(self):
        max_bytes = self.max_mb * 1024 * 1024
        while self.current_memory > max_bytes and self.usage_order:
            # Remove least recently used file
            file_to_remove = self.usage_order.pop(0)
            data_dict = self.cache.pop(file_to_remove)
            freed = data_dict['x'].nbytes + data_dict['y'].nbytes
            self.current_memory -= freed