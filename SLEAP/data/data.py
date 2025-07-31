from data.signal import Signal
from Globals import DataManager

class Data:
    DATASET = "sleep-EDF-20"

    @staticmethod
    def get_dataset_path_by_signal(signal: Signal):
        return f"data/{Data.DATASET}/{signal.name}"
    
    @staticmethod
    def set_dataset(dataset):
        if dataset not in ("sleep-EDF-20", "sleep-EDF-78", "sleep-EDFx"):
            raise ValueError(f"Dataset does not exist: {dataset}.")
        Data.DATASET = dataset
    
    @staticmethod
    def set_data_percentage(percent):
        DataManager.dataset_percentage = percent

