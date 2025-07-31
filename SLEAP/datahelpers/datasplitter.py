import os
import numpy as np
from datahelpers.data import Data

da = Data()
dataset_name = da.find_dataset()

for signal in da.get_all_signal_names():
    data_file_name_within_signal = os.listdir(f"data/{dataset_name}/{signal}")[0]

    with np.load(f"data/{dataset_name}/{signal}/{data_file_name_within_signal}") as huge_file:
        rows = huge_file.files