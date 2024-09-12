import numpy as np

def min_max_normalize(min_val, max_val, value):
    if max_val == min_val:
        # Avoid division by zero if min_val equals max_val
        return 0 if value == min_val else np.nan
    return (value - min_val) / (max_val - min_val)
