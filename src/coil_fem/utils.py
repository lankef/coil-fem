from jax.nn import sigmoid

def fetch_attr(func_name: str, module):
    if hasattr(module, func_name):
        attr = getattr(module, func_name)
    else:
        raise ValueError(f'\'{func_name}\' not found in {str(module)}')
    return attr

def clamp_sigmoid(distances_sq, r_clamp, sigmoid_eps):
    sigmoid_width = sigmoid_eps * r_clamp
    return sigmoid((r_clamp**2 - distances_sq) / (sigmoid_width**2))