import jax.numpy as jnp
from .supports import Support

class EulerNetworkSupport(Support):
    """A class that stores a support structure formed by a network of Euler beams.

    EulerNetworkSupport models a support cage using a network 
    of Euler-Bernoulli beams. There are two types of beams: 
    1. Coil-coil (CC) beams. These beams connects two adjacent coils.
    2. Coil-foundation (CF) beams. These beams connects a coil to the foundation.
    
    """

    def __init__(n_beams_cc, n_beams_cf):
        self.

        