import numpy as np
import numpy.typing as npt
import scipy

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from helpers import *
from constants import *

from typing import List, Tuple, Callable


class CentralEquation:
    def __init__(self, basis_vectors: float|npt.NDArray, atoms: npt.NDArray, atom_potential: Callable, num_cells: int, res_cells: int):
        """
        Args:
            a: basis vectors
            b: basis vectors in reciprocal space

            r_unit: grid of the unit cell (with res_cells number of points)
            g: reciprocal lattice (with res_cells number of points)
        """


        # read in basis vectors
        if isinstance(basis_vectors, (int, float)):
            self.a = np.array([[basis_vectors]])
            self.dim = 1
        else:
            self.a = np.array(basis_vectors)
            assert(self.a.shape[0] == self.a.shape[1]) # a should be square to be a basis
            assert(self.a.shape[0] <= 3) # dimension should be at maximum 3
            self.dim = self.a.shape[0]

        # read in lattice points
        self.lattice_points = np.einsum('ij,...i->...j',self.a, atoms)

        # dimension has to match
        if self.dim == 1:
            assert(len(self.lattice_points.shape) == 1)
        else:
            assert(self.lattice_points.shape[1] == self.dim)

        # generate reciprocal basis vectors
        if self.dim == 1:
            self.b = 2*np.pi / self.a
        elif self.dim == 2:
            n = np.roll(self.a, 1, axis=(0, 1))
            n[:, 1] *= -1
            self.b = 2*np.pi * n / (np.vecdot(self.a, n))[:, np.newaxis]
        else:
            n = np.cross(np.roll(self.a, -1, axis=0), np.roll(self.a, -2, axis=0))
            self.b = 2*np.pi * n / (np.vecdot(self.a, n))[:, np.newaxis]

        # read in number and resolution
        self.num_cells = num_cells
        self.res_cells = res_cells

        # read in atom potential
        self.atom_potential = atom_potential

        # generate grids
        self.r_unit, self.r_unit_idx = create_lattice(1, self.res_cells, self.a)
        self.g, self.g_idx = create_lattice(self.res_cells, self.res_cells, self.b)
        self.k_unit, self.k_unit_idx = create_lattice(1, self.num_cells, self.b)

        # create potential
        self.V = np.sum(self.atom_potential(self.r_unit[:,:,:, np.newaxis] - self.lattice_points), axis=3)
        self.V = np.nan_to_num(self.V, posinf=np.max(self.V[~np.isinf(self.V)]), neginf=np.min(self.V[~np.isneginf(self.V)])) # remove inf and -inf values

        # calculate fft
        self.U = np.fft.fftshift(np.fft.fftn(self.V))

    def solve_central_equation(self, k):
        """
        Solves the central equation for the wavevectors k

        Args:
            k: Wavevectors (crystal momentum quantum numbers) for which the central equation should be solved for

        Returns: Returns the fourier coefficients and energies
            E: shape is (len(k), res_cells^dim)
            C: shape is (len(k), res_cells^dim, res_cells^dim)
        """
        # build system of equations from central equations
        T = h_bar**2/(2*m_e) * np.linalg.norm(k[:, np.newaxis, np.newaxis, np.newaxis] - self.g, axis=-1)**2 # in
        T = np.reshape(T, (len(k), self.res_cells**self.dim))
        A = 0.001*scipy.linalg.circulant(np.ravel(self.U)) + np.apply_along_axis(np.diag, 1, T)

        # find energies and coefficients for all k (shape (k, n))
        self.E, self.C = np.linalg.eigh(A)

        # sort energies
        perm = np.argsort(np.mean(self.E, axis=0))
        self.E = self.E[:, perm]
        self.C = self.C[:, perm]

    def wignerseitz_cell(self):
        """
        Calculate the Wigner-Seitz cell

        Returns: Returns the vertices, mid-edge points and mid-face points of the Wigner-Seitz cell
            verts: shape is (ridges, *)
            edges: shape is (ridges) if 2d, (ridges, *) if 3d
            faces: shape is (ridges) if 3d
        """
        # flatten reciprocal lattice
        recp_lattice = np.reshape(self.g, (-1, self.dim))
        indices = np.reshape(self.g_idx, (-1, self.dim))

        # compute voronoi
        vor = scipy.spatial.Voronoi(recp_lattice)

        # find all ridges confining the wigner seitz cell
        central_point_idx = np.argwhere(np.linalg.norm(indices, axis=1) == 0)[0]
        ridges_idx = np.argwhere(np.bitwise_or.reduce(np.isin(vor.ridge_points, central_point_idx), axis=1)).ravel()
        ridges = [vor.ridge_vertices[i] for i in ridges_idx]
        self.ws_verts = [[vor.vertices[i] for i in r] for r in ridges] # lists are not rectangular!

        # compute mid edges and mid faces
        if self.dim == 1:
            pass
        elif self.dim == 2:
            edges = [np.mean(r, axis=0) for r in self.ws_verts]
            return self.ws_verts, edges
        else:
            edges = [[(r[i] + r[(i+1) % len(r)]) / 2 for i in range(len(r))] for r in self.ws_verts]
            faces = [np.mean(r, axis=0) for r in self.ws_verts]
            return self.ws_verts, edges, faces

    def plot_wignerseitz_cell(self, show_symmetries: bool = True, ax: object = None):
        """
        Plot the Wigner-Seitz cell

        Args:
            show_symmetries: If true, the specified symmetry wave vectors are plotted as well
            ax: Matplotlib axes object
        """
        # get axes if none were specified
        if ax is None:
            ax = plt.gca()

        # plot
        if self.dim == 1:
            pass
        elif self.dim == 2:
            lines = mpl.collections.LineCollection(self.ws_verts, color='black')
            ax.add_collection(lines)
        else:
            polys = Poly3DCollection(self.ws_verts, ec='black', alpha=0)
            ax.add_collection(polys)

        if show_symmetries:
            for label, pos in self.symmetries.values():
                ax.text(*pos, label, va='bottom', ha='left')

                if np.linalg.norm(pos) < 1e-6:
                    ax.scatter(*pos, color='black')
                else:
                    if self.dim == 1:
                        pass
                    elif self.dim == 2:
                        ax.quiver(0, 0, *pos, scale=1, angles='xy', units='dots', scale_units='xy', color='black')
                    else:
                        arrow = Arrow3D(*zip([0, 0, 0], pos), mutation_scale=10, arrowstyle='-|>', color='black')
                        ax.add_artist(arrow)
        
    def register_symmetries(self, symmetries: dict):
        """
        Adds the symmetry wavevectors

        Args:
            symmetries: A dict specifing the symmetry wavevectors (crystal momentum quantum numbers). The dict has the shape {'name': ('label', wavevector), ... }
        """
        self.symmetries = symmetries

    def build_symmetry_path(self, symmetries, res):
        """
        Creates a path, interpolating linearly between the symmetry wavevectors

        Args:
            symmetries: All the symmetries the path should visit.
            res: The number of points between each symmetry wavevector

        Returns: Returns the path and the ticks for plotting
            path: shape is ((len(symmetries)-1)*res)
            ticks: the numbers at which ticks should be placed
            tick_labels: the labels of each tick
        """
        path = np.empty((0, self.dim))
        ticks = np.arange(len(symmetries))*res
        tick_labels = []

        for i in range(len(symmetries)-1):
            k1 = self.symmetries[symmetries[i]][1]
            k2 = self.symmetries[symmetries[i+1]][1]
            path = np.append(path, np.linspace(k1, k2, num=res), axis=0)
            
            tick_labels.append(self.symmetries[symmetries[i]][0])
        tick_labels.append(self.symmetries[symmetries[-1]][0])
        return path, ticks, tick_labels





