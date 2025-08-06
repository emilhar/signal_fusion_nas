import os
from datahelpers.signal import Signal
from datahelpers.target import Target
from Globals import Globals

class Data:
    __ALL_SIGNAL_NAMES = None
    __ALL_TARGET_NAMES = None
    DIRECTORY = "dataset"
    max_memory = Globals.lazy_data_max_memory
    batch_size = 128
    

    def __init__(self):
        self.dataset = Data.find_dataset()
        self.signal_objects = self.__create_signals_from_dataset()
        self.target_objects = self.__create_targets()

    def __create_signals_from_dataset(self):
        signals = []
        data_dir = f"{Data.DIRECTORY}/{self.dataset}"
        for signal_name in os.listdir(data_dir):
            new_signal = Signal(signal_name, f"{data_dir}/{signal_name}")
            signals.append(new_signal)
        
        return signals
    
    def __create_targets(self) -> list[Target]:
        targets = []

        with open(f'{Data.DIRECTORY}/label_map.txt', 'r') as file:
            for line in file:
                given_name, data_label = line.strip().split()
                given_name = given_name.removesuffix(":")
                targets.append(Target(given_name, int(data_label)))

        return targets
    
    def get_stage_map(self, classification_class:Target):
        stage_map = {}
        for target in self.target_objects:
            stage_map[target.data_label] = 1 if classification_class.given_name == target.given_name else 0

        return stage_map

    @staticmethod
    def get_all_signal_names():
        if Data.__ALL_SIGNAL_NAMES is None:
            ds = Data.find_dataset()
            Data.__ALL_SIGNAL_NAMES = os.listdir(f"./{Data.DIRECTORY}/{ds}")

        return Data.__ALL_SIGNAL_NAMES

    @staticmethod
    def get_all_target_names():

        def format_name(line:str):
            name, _ = line.strip().split()
            name = name.removesuffix(":")
            return name
        
        if Data.__ALL_TARGET_NAMES is None:
            with open(f'{Data.DIRECTORY}/label_map.txt', 'r') as file:
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
        dataset = [filename for filename in os.listdir(f"{Data.DIRECTORY}") if __datafilter(filename)]
        if not dataset:
            raise FileNotFoundError("Could not find dataset, please see README")
        
        return dataset[0] 
