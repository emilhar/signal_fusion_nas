"""
This file contains the Manager needed to make a model
"""

def get_branch_configs(branches:list[list[int]], name:str, sample_count:int):
  """ 
    Get full Manager for a model given
      left_kernel_size: A list of 3 numbers for the kernel sizes of the left branch. e.g. [1, 2, 3]
      right_kernel_size: A list of 3 numbers for the kernel sizes of the right branch. e.g. [1, 2, 3]
      name: The name of the model. e.g. "MyN3Classifier
      sample_count = number with number of samples

    Outputs model args for model
  """
  conv_stride, pool_size, pool_strides = _find_pool_sizes(sample_count)

  branch_configs = {}

  for i, branch in enumerate(branches):

      branch_configs[f"branch_{i}"] = {
          "num_kernels": [32, 64, 64],
          #"num_kernels": [16, 32, 32],
          "kernel_sizes": branch,
          "paddings": _kernel_to_pad(branch),
          "strides": [conv_stride, 1, 1],
          "pool_sizes": [pool_size, pool_size//2],
          "pool_strides": [pool_strides, pool_strides//2],
          "dropout_rates": [0.1, 0.0]
      }
  

  model_args = {
    "name": name,
    "n_samples": sample_count,
    "branch_configs": branch_configs
    }
  
  return model_args


def _kernel_to_pad(numbers: list[int]):
  """Takes a kernel_sizes list and returns a corresponding paddings list"""
  new_list = []

  for num in numbers:
    fixed_num = (num // 2) - 1
  
    if fixed_num < 0:
      fixed_num = 0

    new_list.append(fixed_num)

  return new_list

def _find_pool_sizes(n_samples: int):
    
    conv_stride = max(n_samples // 30 // 16, 1)
    pool_size = max(n_samples // 30 // 12, 1)
    pool_strides = max(pool_size // 2, 1)

    return conv_stride, pool_size, pool_strides
