"""Utils for model executor."""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)


def pad_weight(weight: torch.Tensor, num_pad: int, pad_dim: int = 0):  
    if weight.dim() == 1:  
        padding = torch.zeros(num_pad, dtype=weight.dtype, device=weight.device)  
        padded_weight = torch.cat([weight, padding], dim=0)  
    elif weight.dim() == 2:   
        if pad_dim == 0:  
            padding = torch.zeros(num_pad, weight.shape[1], dtype=weight.dtype, device=weight.device)  
            padded_weight = torch.cat([weight, padding], dim=0)  
        elif pad_dim == 1:  
            padding = torch.zeros(weight.shape[0], num_pad, dtype=weight.dtype, device=weight.device)  
            padded_weight = torch.cat([weight, padding], dim=1)  
        else:  
            raise ValueError("pad_dim must be 0 or 1")  
    else:  
        raise ValueError("Weight tensor must be 1D or 2D")   
    padded_weight = padded_weight.contiguous()
    return padded_weight  


def gemm_bank_conf(weight):  
    is_mul_of_2048 = weight % 2048 == 0     
    is_power_of_two = (weight & (weight - 1)) == 0 and weight != 0  
      
    if is_mul_of_2048 and is_power_of_two:  
        return True 
    else:  
        return False  