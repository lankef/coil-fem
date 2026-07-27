from jax.nn import sigmoid


def clamp_sigmoid(d_sq, r, sigmoid_eps):
    sigmoid_width = sigmoid_eps * r
    return sigmoid((r**2 - d_sq) / (sigmoid_width**2))