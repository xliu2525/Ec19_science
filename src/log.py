import inspect
import re
import numpy as np
import torch
import jax.numpy as jnp

log_values_cache = {}

def desc_obj(obj):
    if isinstance(obj, torch.Tensor):
        return f"tensor shape {obj.shape}, dtype {obj.dtype}"
    elif isinstance(obj, jnp.ndarray):
        return f"Jax shape {obj.shape}, dtype {obj.dtype}"
    elif isinstance(obj, np.ndarray):
        return f"shape {obj.shape}, dtype {obj.dtype}"
    elif isinstance(obj, list):
        return f"len: {len(obj)}" + f" First elem: {desc_obj(obj[0])}" if len(obj) > 0 else ""
    elif isinstance(obj, dict):
        return f"keys: {list(obj.keys())}"
    else:        
        return f"{obj}"

def dlog(*args, v=0, s=1):
    stack = inspect.stack()
    frame = stack[1]
    code_context = frame.code_context[0]
    arguments_match = re.search("dlog\((.*)\)", code_context)
    if not arguments_match:
        print(f"Couldn't parse arguments in {code_context}")
        return
    
    args_split_match = re.split(",\s*(?![^()[\]]*[)\]])", arguments_match.groups(1)[0])

    for (i, arg) in enumerate(args):
        name = args_split_match[i]
        if s:
            log_values_cache[name] = arg
        if name == f"\"{arg}\"":
            print(f"{arg}:")
        else:
            print(f"{name}: {desc_obj(arg)}")
            if v:
                print(arg)
                
def dget():
    stack = inspect.stack()
    frame = stack[1]
    code_context = frame.code_context[0]
    arguments_match = re.search("(.*?)\s*=\s*dget", code_context)
    if not arguments_match:
        print(f"Couldn't parse arguments in {code_context}")
        return
    
    args_split_match = re.split(",\s*(?![^()[\]]*[)\]])", arguments_match.groups(1)[0])
    values = [log_values_cache[arg_name] for arg_name in args_split_match]
    return tuple(values) if len(values) > 1 else values[0]
    