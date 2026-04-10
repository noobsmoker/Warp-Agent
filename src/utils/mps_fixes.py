"""
MPS Backend Stability Fixes for Apple Silicon

Addresses:
- Non-contiguous tensor failures
- Adam optimizer state issues
- Limited operator support
"""

import torch
import torch.nn as nn
from typing import Callable


def enforce_contiguous_tensors(model: nn.Module) -> nn.Module:
    """
    Wrap model to ensure all outputs are contiguous.
    MPS fails silently with non-contiguous tensors.
    """
    original_forward = model.forward
    
    def contiguous_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if isinstance(result, torch.Tensor) and not result.is_contiguous():
            result = result.contiguous()
        elif isinstance(result, tuple):
            result = tuple(
                r.contiguous() if isinstance(r, torch.Tensor) and not r.is_contiguous() else r
                for r in result
            )
        return result
    
    model.forward = contiguous_forward
    return model


def patch_mps_operations():
    """Patch known MPS operation bugs."""
    # Force contiguous format for Adam optimizer states
    import torch.optim.adam as adam_module
    original_adam_init = adam_module.Adam.__init__
    
    def patched_adam_init(self, params, **kwargs):
        # Ensure all params are contiguous before optimizer init
        contiguous_params = []
        for p in params:
            if p is not None and hasattr(p, 'data'):
                if not p.data.is_contiguous():
                    p.data = p.data.contiguous()
                contiguous_params.append(p)
            else:
                contiguous_params.append(p)
        return original_adam_init(self, contiguous_params, **kwargs)
    
    adam_module.Adam.__init__ = patched_adam_init


def safe_mps_operation(operation: Callable) -> Callable:
    """
    Decorator to ensure MPS operations are safe.
    """
    def wrapper(*args, **kwargs):
        # Ensure all tensor inputs are contiguous
        safe_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                arg = arg.contiguous()
            safe_args.append(arg)
        
        safe_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor) and not val.is_contiguous():
                val = val.contiguous()
            safe_kwargs[key] = val
        
        result = operation(*safe_args, **safe_kwargs)
        
        # Ensure output is contiguous
        if isinstance(result, torch.Tensor) and not result.is_contiguous():
            result = result.contiguous()
        elif isinstance(result, tuple):
            result = tuple(
                r.contiguous() if isinstance(r, torch.Tensor) and not r.is_contiguous() else r
                for r in result
            )
        
        return result
    
    return wrapper


def get_safe_device() -> torch.device:
    """Get the safest device for current hardware."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def mps_memory_cleanup():
    """Clear MPS cache to free memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()