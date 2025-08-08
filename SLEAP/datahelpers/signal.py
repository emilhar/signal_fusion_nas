import os
import numpy as np

class Signal:
    def __init__(self, name: str, signal_directory:str):
        self.name = name

        example_file_from_signal = os.listdir(signal_directory)[0] #TODO: Change to 0, this is 1 because of gitignore
        full_file_path = f"{signal_directory}/{example_file_from_signal}"
        
        with np.load(full_file_path, allow_pickle=True) as data:
            first_array = data[data.files[0]]
            self.n_samples = first_array.shape[1]

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name