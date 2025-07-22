import random
import time
import math

import numpy as np
import torch

from Globals import Signal, Sleepstage, DataManager, ModelManager
from EAController.SleepDataLoader import SleepDataLoader
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs

class KRNL_GridSearch:
    """
    Kernel, Reduction function, N-Layer grid search
    """
    def __init__(self, signal: Signal, sleep_stage: Sleepstage, dataset: DataManager.DatasetNames, dataset_percentage=0.10, n_samples=3000):
        DataManager.MAX_MEMORY = 2048
        DataManager.DATASET = dataset
        DataManager.dataset_percentage = dataset_percentage
        ModelManager.BATCH_SIZE = 32

        self.kernels = self.__get_kernels()
        self.layers = self.__get_layers()
        self.reduction_to_name = None
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


    def theta(self, n: int) -> list[list[int]]:
        opt = [[19, 18], [420, 120, 8], [1000, 1000, 1000], [1, 1, 1], [1000], [900, 500, 500]]

        individuals = []
        for _ in range(n):
            k = random.choice(opt)
            for i, x in enumerate(k):
                k[i] *= random.uniform(0.5, 1.5)
            random.shuffle(k)
            individuals.append(k)

        return individuals
    
    def compute_grid(self):
        start_time = time.time()
        total_iterations = len(self.kernels) * len(self.reductions) * len(self.layers)
        completed = 0
        
        self.grid = np.empty((len(self.kernels), len(self.reductions), len(self.layers)), dtype=dict)
        for x, kernel in enumerate(self.kernels):
            for y, reduction in enumerate(self.reductions):
                for z, layer in enumerate(self.layers):
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
        train_loader, test_loader, n_samples, pos_weight = self.loader.get_random_subset() 

        branch = [kernel]
        for _ in range(layer-1):
            kernel = reduction(kernel)
            branch.append(kernel)


        model_args = get_branch_configs([branch], self.n_samples)
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
        k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k += [12, 14, 16, 18, 20]
        k += [22, 24, 26, 28, 30]
        k += [35, 40, 50]
        k += [60, 70, 80, 90, 100]
        k += [120, 140, 160, 180]
        k += [200, 250, 300, 350, 400]
        k += [500, 600, 700, 800, 900, 1000]
        k += [1250, 1500]

        return k
    
    def __get_layers(self) -> list[int]:
        return [1, 2, 3, 4]
    
    def __get_reductions(self) -> list:
        def identity(x):
            return x
        
        def halve(x):
            return max(1, x//2) 
        
        def rooting(x):
            return max(1, int(x**0.5))

        def log2(x):
            return max(1, int(math.log2(x))) if x > 1 else 1
        
        def divide5(x):
            return max(1, x//5)
        

        
        self.reduction_to_name = {
            identity: "identity",
            halve: "halve",
            rooting: "rooting",
            log2: "log2",
            divide5: "divide5"
        }
        
        return [identity, halve, rooting, log2, divide5]

    
import time
import math
import os
import json
import numpy as np
import torch
from skopt import Optimizer
from skopt.space import Integer, Categorical

from Globals import Signal, Sleepstage, DataManager, ModelManager
from EAController.SleepDataLoader import SleepDataLoader
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs



class KRNL_BayesianSearch:
    """
    Bayesian optimization for kernel sizes (no reduction functions).
    """
    def __init__(self, signal: Signal, sleep_stage: Sleepstage, dataset: DataManager.DatasetNames, dataset_percentage=0.10, n_samples=3000, n_calls=50):
        DataManager.MAX_MEMORY = 2048
        DataManager.DATASET = dataset
        DataManager.dataset_percentage = dataset_percentage
        ModelManager.BATCH_SIZE = 32

        self.signal = signal
        self.sleep_stage = sleep_stage
        self.n_samples = n_samples
        self.n_calls = n_calls
        self.loader = SleepDataLoader(self.signal, self.sleep_stage)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load fixed dataset subset for consistent evaluation
        self.train_loader, self.test_loader, _, self.pos_weight = self.loader.get_random_subset()
        
        # Define search space: number of layers and kernel sizes for up to 4 layers
        kernels = self.__get_kernels()
        self.space = [
            Integer(1, 4, name='n_layers'),
            Categorical(kernels, name='kernel1'),
            Categorical(kernels, name='kernel2'),
            Categorical(kernels, name='kernel3'),
            Categorical(kernels, name='kernel4'),
        ]
        
        # Bayesian optimizer
        self.optimizer = Optimizer(
            self.space,
            base_estimator="GP",
            n_initial_points=min(10, n_calls),
            acq_func="EI"
        )
        
        self.history = []

def build_branch(self, params):
    """Construct branch from parameters"""
    n_layers = params[0]
    kernels = [int(k) for k in params[1:1+n_layers]]  # Convert back to int
    return kernels

    def evaluate_branch(self, branch):
        """Train and evaluate model with given branch."""
        model_args = get_branch_configs([branch], self.n_samples)
        model_args["batch_size"] = 32
        model = CNN_BinaryClassifier(**model_args).to(self.device)
        
        model_performance = train_model(
            model,
            self.device,
            self.train_loader,
            self.test_loader,
            self.pos_weight,
            epochs=1
        )
        
        return {
            k: v for k, v in model_performance.items()
            if k not in ["true_labels", "best_scores", "state_dict"]
        }

    def run_optimization(self):
        """Execute Bayesian optimization."""
        print(f"Starting Bayesian optimization for {self.n_calls} iterations...")
        start_time = time.time()
        
        for i in range(self.n_calls):
            iter_start = time.time()
            
            # Get next hyperparameters from optimizer
            params = self.optimizer.ask()
            
            # Build branch from parameters
            branch = self.build_branch(params)
            result = self.evaluate_branch(branch)
            train_loss = result['train_loss']
            
            # Store results and update optimizer
            self.history.append({
                'params': params,
                'branch': branch,
                'result': result
            })
            self.optimizer.tell(params, train_loss)  # Minimize train loss
            
            # Log progress
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time
            print(f"Iter {i+1}/{self.n_calls}: train loss={train_loss:.4f}, "
                  f"Branch={list(map(int, branch))}, Time={iter_time:.1f}s, Total={total_time:.1f}s")
        
        # Save results
        self.save_results()
        print("Optimization completed.")

    def save_results(self):
        """Save optimization history to JSON file."""
        results_dir = "./Data/bayesian_results/"
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{results_dir}{self.signal}_{self.sleep_stage}.json"
        
        with open(filename, 'w') as f:
            json.dump(list(map(int, self.history)), f, indent=4)
        print(f"Results saved to {filename}")

    def __get_kernels(self) -> list[int]:
        k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k += [12, 14, 16, 18, 20]
        k += [22, 24, 26, 28, 30]
        k += [35, 40, 50]
        k += [60, 70, 80, 90, 100]
        k += [120, 140, 160, 180]
        k += [200, 250, 300, 350, 400]
        k += [500, 600, 700, 800, 900, 1000]
        k += [1250, 1500]

        return k
    