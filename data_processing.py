import numpy as np

def min_max_normalize(min_val, max_val, value):
    if max_val == min_val:
        # Avoid division by zero if min_val equals max_val
        return 0 if value == min_val else np.nan
    return (value - min_val) / (max_val - min_val)

# def min_max_normalize(min_val, max_val, value):
#     if max_val == min_val:  # Avoid division by zero
#         return 0.5  # Or any other value or behavior you prefer in this case
#     return (value - min_val) / (max_val - min_val)