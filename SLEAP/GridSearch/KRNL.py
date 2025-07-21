import random
import time

import numpy as np
import torch

from Globals import Signal, Sleepstage, DataManager, Clr
from EAController.SleepDataLoader import SleepDataLoader
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs

class KRNL_GridSearch:
    """
    Kernel, Reduction function, N-Layer grid search
    """
    def __init__(self, signal: Signal, sleep_stage: Sleepstage, n_samples=3000):
        DataManager.MAX_MEMORY = 2048
        DataManager.DATASET = DataManager.DatasetNames.EDF_78

        self.kernels = self.__get_kernels()
        self.layers = self.__get_layers()
        self.reductions = self.__get_reductions()
        self.signal = signal
        self.sleep_stage = sleep_stage
        self.n_samples = n_samples
        self.loader = SleepDataLoader(self.signal, self.sleep_stage)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.precomputed():
            print("Computing KRNL grid...")
            self.compute_grid()
            np.save(f"./Data/grids/{self.signal}/{self.sleep_stage}", self.grid)

        print("Finished loading grid.")


    def theta(self, n: int) -> list[CNN_BinaryClassifier]:
        return ...
    
    def compute_grid(self):
        start_time = time.time()
        total_iterations = len(self.kernels) * len(self.layers) * len(self.reductions)
        completed = 0
        
        self.grid = np.empty((len(self.kernels), len(self.layers), len(self.reductions)), dtype=dict)
        for x, kernel in enumerate(self.kernels):
            for z, reduction in enumerate(self.reductions):
                for y, layer in enumerate(self.layers):
                    self.grid[x, y, z] = self.new_model(kernel, layer, reduction)
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    percent = (completed / total_iterations) * 100
                    
                    if completed > 0:
                        iter_time = elapsed / completed
                        remaining = max(0, (total_iterations - completed) * iter_time)
                        eta_sec = int(remaining)
                        eta_str = f"{eta_sec//3600:02d}:{eta_sec%3600//60:02d}:{eta_sec%60:02d}"
                    else:
                        eta_str = "--:--:--"
                    
                    print(f"\rProgress: [{'='*int(percent/5)}{' '*(20-int(percent/5))}] {percent:.1f}% • ETA: {eta_str}", end="\r")
        
        total_time = time.time() - start_time
        print(f"\nCompleted {total_iterations} models in {total_time:.1f} seconds")


    
    def precomputed(self) -> bool:
        try:
            self.grid = np.load(f"./Data/grids/{self.signal}/{self.sleep_stage}.npy", allow_pickle=True)
            print("Grid already exists in ./Data/grids/")
            return True
            
        except FileNotFoundError as e:
            print("Grid has not been precomputed.")
            return False
    
    
    def new_model(self, kernel: int, layer: int, reduction):
        train_loader, test_loader, n_samples, pos_weight = self.loader.get_random_subset(
            dataset_percentage=0.05,
            batch_size=32,
        ) 

        branch = [kernel]
        for _ in range(layer-1):
            kernel = reduction(kernel)
            branch.append(kernel)


        model_args = get_branch_configs([branch], "", self.n_samples)
        model_args["batch_size"] = 32
        model = CNN_BinaryClassifier(**model_args).to(self.device)
        
        model_performance = train_model(
            model,
            self.device,
            train_loader,
            test_loader,
            pos_weight,
            epochs=1,
        )

        model_performance = {
            k: v for k, v in model_performance.items()
            if k != "true_labels" and k != "best_scores" and k != "state_dict"
        }

        return model_performance

    
    def __get_kernels(self) -> list[int]:
        return [x for x in range(1, 501, 10)]
    
    def __get_layers(self) -> list[int]:
        return [1, 2, 3, 4]
    
    def __get_reductions(self) -> list:
        def identity(x):
            return x
        
        def halve(x):
            return max(1, x//2) 
        
        return [identity, halve]

    

