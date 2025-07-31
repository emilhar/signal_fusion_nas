import os
from datahelpers.signal import Signal
from datahelpers.target import Target
import Globals

class Data:
    __ALL_SIGNAL_NAMES = None
    __ALL_TARGET_NAMES = None

    def __init__(self):
        self.dataset = Data.find_dataset()
        self.signal_objects = self.__create_signals_from_dataset()
        self.target_objects = self.__create_targets()

    def __create_signals_from_dataset(self):
        signals = []
        data_dir = f"data/{self.dataset}"
        for signal_name in os.listdir(data_dir):
            new_signal = Signal(signal_name, f"{data_dir}/{signal_name}")
            signals.append(new_signal)
        
        return signals
    
    def __create_targets(self):
        targets = []
        with open('data/label_map.txt', 'r') as file:
            for line in file:
                given_name, data_label = line.strip().split()
                given_name = given_name.removesuffix(":")
                targets.append(Target(given_name, data_label))

        Globals._target_objs = targets

        return targets

    @staticmethod
    def get_all_signal_names():
        if Data.__ALL_SIGNAL_NAMES is None:
            ds = Data.find_dataset()
            Data.__ALL_SIGNAL_NAMES = os.listdir(f"./data/{ds}")

        return Data.__ALL_SIGNAL_NAMES

    @staticmethod
    def get_all_target_names():

        def format_name(line:str):
            name, _ = line.strip().split()
            name = name.removesuffix(":")
            return name
        
        if Data.__ALL_TARGET_NAMES is None:
            with open('data/label_map.txt', 'r') as file:
                Data.__ALL_TARGET_NAMES = [format_name(line) for line in file]

        return Data.__ALL_TARGET_NAMES

    @staticmethod
    def find_dataset():

        # Helpful function to work through the data folder with
        def __datafilter(filename:str):
            if filename in ["label_map.txt", "__pychache__"]:
                return False
            return True

        # List of everything inside the data folder, except for the label map and pycache
        dataset = [filename for filename in os.listdir("data") if __datafilter(filename)]
        if not dataset:
            raise FileNotFoundError("Could not find dataset, please see README")
        
        return dataset[0] 
