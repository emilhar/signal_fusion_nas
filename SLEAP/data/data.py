from data.signal import Signal

class Data:
    DATASET = "sleep-EDF-20"

    @staticmethod
    def get_dataset_path_by_signal(signal: Signal):
        return f"data/{Data.DATASET}/{signal.name}"
    
    
