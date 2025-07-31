# Base Imports
import torch
from torch.utils.data import DataLoader

# Model and Training imports
from models.cnn_binary_classifier import CNN_BinaryClassifier
from Globals import EvolutionManager, LoggingHelper, device

class TrainedModelMaker:

    def __init__(
            self, 
            branches: list[list[int]],
            N_SAMPLES: int, 
            pos_weight: torch.FloatTensor, 
            train_loader: DataLoader, 
            test_loader: DataLoader,
            epochs,
            batch_size,
            filters=32
        ):

        assert device.type == "cuda", f"WHAT: {device.type}"

        self.model_args = self.get_branch_configs(branches, N_SAMPLES, filters) # See ModelManager
        self.model_args["batch_size"] = batch_size
        self.model = CNN_BinaryClassifier(**self.model_args).to( device )

        if EvolutionManager.VERY_VERBOSE:
            print(f"\n\nTraining model: {branches=}, Generation: {LoggingHelper.current_generation_id}/{EvolutionManager.GENERATIONS}, Generation Completeness: {LoggingHelper.current_individual_id}/{LoggingHelper.population_size}")

        self.model_performance = CNN_BinaryClassifier.train_model(
            self.model, 
            train_loader, 
            test_loader, 
            pos_weight, 
            verbose=EvolutionManager.VERY_VERBOSE, 
            epochs=epochs,
        )

    def get_branch_configs(self, branches:list[list[int]], sample_count:int, filters):
        """ 
        Get full Manager for a model given
        left_kernel_size: A list of 3 numbers for the kernel sizes of the left branch. e.g. [1, 2, 3]
        right_kernel_size: A list of 3 numbers for the kernel sizes of the right branch. e.g. [1, 2, 3]
        name: The name of the model. e.g. "MyN3Classifier
        sample_count = number with number of samples

        Outputs model args for model
        """
        conv_stride, pool_size, pool_strides = self.__find_pool_sizes(sample_count)

        branch_configs = {}

        for i, branch in enumerate(branches):
            branch_configs[f"branch_{i}"] = {
                "num_kernels": self.get_num_kernels(filters, branch),
                "kernel_sizes": branch,
                "paddings": self.__kernel_to_pad(branch),
                "strides": self.__get_strides(branch, sample_count),
                "pool_sizes": self.__get_pool_sizes(branch, sample_count),
                "pool_strides": self.__get_pool_strides(branch, sample_count),
                "dropout_rates": [0.0] + [0.0] * (len(branch)-1)
            }

        model_args = {
            "n_samples": sample_count,
            "branch_configs": branch_configs
        }
  
        return model_args

    def __clamp_num(self, num):
        return max(1, num//2)

    def __kernel_to_pad(self, numbers: list[int]):
        """Takes a kernel_sizes list and returns a corresponding paddings list"""
        new_list = []

        for num in numbers:
            fixed_num = (num // 2) - 1
            if fixed_num < 0:
                fixed_num = 0
            new_list.append(fixed_num)

        return new_list

    def __find_pool_sizes(self, n_samples: int):
        
        conv_stride = max(n_samples // 30 // 16, 1)
        pool_size = max(n_samples // 30 // 12, 1)
        pool_strides = max(pool_size // 2, 1)

        return conv_stride, pool_size, pool_strides

    def get_num_kernels(self, filters, branch):
        return [filters] * len(branch)

    def __get_strides(self, branch, n_samples):
        conv_stride = max(n_samples // 30 // 16, 1)
        return [conv_stride] + [1]*(len(branch)-1)

    def __get_pool_sizes(self, branch, n_samples):
        pool_size = max(n_samples // 30 // 12, 1)
        return [pool_size] + [self.__clamp_num(pool_size)] * (len(branch)-1)

    def __get_pool_strides(self, branch, n_samples):
        pool_stride = max(max(n_samples // 30 // 12, 1) // 2, 1)
        return [pool_stride] + [self.__clamp_num(pool_stride)] * (len(branch)-1)