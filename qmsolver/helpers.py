import numpy as np
import numpy.typing as npt

import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d

from typing import List, Tuple, Callable

def create_lattice(width: float, num: int, basis: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Creates a lattice in n dimensions. The lattice has the shape (num, ... dim times ..., num, dim). The zero element guaranteed to be in the lattice,
    but the start and stop values are not. The zero element is centered, meaning it has the index floor(num/2) in one axis. Internally the fftfreq and
    fftshift functions are used, to keep the indexing standardized.

    Args:
        width:  greatest-smallest coordinate in one axis
        num:    number of points in one dimension. In total there are num^dim points
        basis:  basis vectors in the shape (dim, dim)
    Returns: the lattice and a array of indices
    """

    # check condition for basis
    assert(basis.shape[0] == basis.shape[1])
    dim = basis.shape[0]

    # create single axis 
    axis = np.fft.fftfreq(num) # [0, ..., floor((num-1)/2), -ceil((num-1)/2), ..., -1] without scaling
    axis_idx = np.arange(num)

    # multiply and stack axis if needed
    if dim == 1:
        grid = axis
        indices = axis_idx
    else:
        axes = np.array(np.meshgrid(*np.tile(axis, reps=(dim, 1))))
        axes_idx = np.array(np.meshgrid(*np.tile(axis_idx, reps=(dim, 1))))
        
        grid =  np.stack(axes.T)
        indices =  np.stack(axes_idx.T)

    # transform the grid
    lattice = np.einsum('ij,...i->...j', basis, grid) * width

    # shift the lattice
    return shift_lattice(lattice), shift_lattice(indices)

def shift_lattice(lattice: npt.NDArray, inverse: bool = False) -> npt.NDArray:
    """
    Shifts the lattice representation from having the zero element centered, to having it at the edge (or back). Internally,
    this function just calls the fftshift function from numpy. We use this function, to keep the indexing of the lattice standardized.

    Args:
        lattice: the lattice, that should be shifted
        inverse: specifies if lattice should be shifted forwards or backwards
    """
    if inverse:
        return np.fft.ifftshift(lattice)
    else:
        return np.fft.fftshift(lattice)
    

def get_coords(array: npt.NDArray) -> Tuple[npt.NDArray]:
    """
    Converts an array of points from the shape (..., dim) to (dim, ...). This can be usefull for plotting

    Args:
        array: the array, this operation should be performed upon
    """
    return np.moveaxis(array, -1, 0)

class Arrow3D(mpl.patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)