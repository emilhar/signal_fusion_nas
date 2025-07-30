import os
import numpy as np
from Globals import DataManager

class Signal:
    __ALL_SIGNALS = None
    def __init__(self, name: str):
        self.name = name
        with np.load(os.listdir(f"./data/{DataManager.DATASET}")[0]) as data:
            self.n_samples = data.shape # IDK


    @staticmethod
    def get_all_signal_names():
        if Signal.__ALL_SIGNALS is None:
            Signal.__ALL_SIGNALS = os.listdir("./data/sleep-EDF-20")
        
        return Signal.__ALL_SIGNALS

