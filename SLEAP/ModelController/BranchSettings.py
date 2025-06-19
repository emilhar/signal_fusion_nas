"""
This file contains the settings needed to make a model
"""

def get_branch_configs(left_kernel_sizes:list[int], right_kernel_sizes:list[int], name:str, sample_count:int):
  """ 
    Get full settings for a model given
      left_kernel_size: A list of 3 numbers for the kernel sizes of the left branch. e.g. [1, 2, 3]
      right_kernel_size: A list of 3 numbers for the kernel sizes of the right branch. e.g. [1, 2, 3]
      name: The name of the model. e.g. "MyN3Classifier
      sample_count = number with number of samples

    Outputs model args for model
  """
  pool_config = _find_pool_sizes(sample_count)

  branch_configs = {
      "left": {
          "num_kernels": [32, 64, 64],
          "kernel_sizes": left_kernel_sizes,
          "paddings": _kernel_to_pad(left_kernel_sizes),
          "strides": [pool_config["left"]["conv_stride1"], 1, 1],
          "pool_sizes": pool_config["left"]["pool_sizes"],
          "pool_strides": pool_config["left"]["pool_strides"],
          "dropout_rates": [0.1, 0.0]
      },
      "right": {
          "num_kernels": [32, 64, 64],
          "kernel_sizes": right_kernel_sizes,
          "paddings": _kernel_to_pad(right_kernel_sizes),
          "strides": [pool_config["right"]["conv_stride1"], 1, 1],
          "pool_sizes": pool_config["right"]["pool_sizes"],
          "pool_strides": pool_config["right"]["pool_strides"],
          "dropout_rates": [0.1, 0.0]
      }
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
    
    left_conv_stride1 = max(n_samples // 30 // 16, 1)
    left_pool_size1 = max(n_samples // 30 // 12, 1)
    left_pool_size2 = max(left_pool_size1 // 2, 1)


    right_conv_stride1 = max(n_samples // 30 // 2, 1)
    right_pool_size1 = max(n_samples // 30 // 24, 1)
    right_pool_size2 = max(right_pool_size1 // 2, 1)

    return {
        "left": {
            "conv_stride1": left_conv_stride1,
            "pool_sizes": [left_pool_size1, left_pool_size2],
            "pool_strides": [left_pool_size1, left_pool_size2]
        },
        "right": {
            "conv_stride1": right_conv_stride1,
            "pool_sizes": [right_pool_size1, right_pool_size2],
            "pool_strides": [right_pool_size1, right_pool_size2]
        }
    }
