# John Eargle (mailto: jeargle at gmail.com)
# onthedl

import numpy as np

__all__ = ["square_comp"]



def square_comp(x, omega, k):
    """
    Single component of square function.

    x:
    omega:
    k:
    """
    return (4.0/np.pi) * np.sin(2*np.pi*(2*k-1)*omega*x)/(2*k-1)
