import scipy
import numpy as np
import matplotlib.pyplot as plt

from qmsolver import constants

class System2D:
    def __init__(self, potential, width, num):
        # member variables
        self.__width = width
        self.__num = num
        self.__step = width / num

        self.__operators = {}
        self.__eigvals = {}
        self.__eigvecs = {}
        self.__basis = np.empty(shape=(self.__num, self.__num))
        self.__eigvals_basis = {}

        self.__potential = potential

        # create helper identity matrix
        I = np.eye(self.__num)

        # create all operators
        s_row = np.linspace(-self.__width/2, self.__width/2, self.__num)
        s_mat = np.diag(s_row)

        self.__operators['x'] = np.kron(I, s_mat)
        self.__operators['y'] = np.kron(s_mat, I)

        d_row = np.zeros(self.__num)
        d_row[-1] = 1 / (2*self.__step); d_row[1] = -1 / (2*self.__step)
        d_mat = scipy.linalg.circulant(d_row)

        self.__operators['dx'] = np.kron(I, d_mat)
        self.__operators['dy'] = np.kron(d_mat, I)

        d2_row = np.zeros(self.__num)
        d2_row[[-1, 1]] = 1 / (self.__step**2); d2_row[0] = -2 / (self.__step**2)
        d2_mat = scipy.linalg.circulant(d2_row)

        self.__operators['dxx'] = np.kron(d2_mat, I)
        self.__operators['dyy'] = np.kron(I, d2_mat)
        self.__operators['lap'] = self.__operators['dxx'] + self.__operators['dyy']

        self.__operators['px'] = -1.0j*constants.hbar * self.op('dx')
        self.__operators['py'] = -1.0j*constants.hbar * self.op('dy')

        self.__operators['H'] = -constants.hbar**2 * self.op('lap') / (2*constants.me) + np.diag(np.ravel(self.__potential))

        # find eigenvalues and eigenvectors of operators
        for op_name, op in self.__operators.items():
            self.__eigvals[op_name], self.__eigvecs[op_name] = self.solve_eigenproblem(op)

    def solve_eigenproblem(self, operator):
        if np.array_equal(np.roll(np.roll(operator, shift=1, axis=0), shift=1, axis=1), operator): # check if operator is circulant
            return np.fft.fft(operator[0]), np.fft.fft(np.eye(self.__num))
        elif scipy.linalg.ishermitian(operator): # check if operator is hermitian
            return scipy.linalg.eigh(operator)
        else:
            return scipy.linalg.eig(operator)

    def op(self, operator):
        if operator in self.__operators:
            return self.__operators[operator]
        else:
            raise ValueError(f'Operator {operator} does not exist!')
        
    def add_operator(self, name, operator, eigenvals=None, eigenvecs=None, override=False):
        if override or name not in self.__operators:
            self.__operators[name] = operator

        if eigenvals is None or eigenvecs is None:
            self.__eigvals[name], self.__eigvecs[name] = self.solve_eigenproblem(operator)
        else:
            self.__eigvals[name], self.__eigvecs[name] = eigenvals, eigenvecs

    def wavepacket(self, x, p, std_x):
        def gaussian(x, mu, sig):
            return np.exp(-np.power((x - mu)/sig, 2)/2)

        state = np.exp(1.0j*p/constants.hbar) * gaussian(self.x, x, std_x)
        
        return self.normalize(state)
    
    def eigenvectors(self, operator):
        if operator in self.__eigvecs:
            return np.moveaxis(self.__eigvecs[operator].reshape(self.__num, self.__num, -1), -1, 0)
        else:
            raise ValueError(f'Operator {operator} does not exist!')
        
    def eigenvalues(self, operator):
        if operator in self.__eigvals:
            return self.__eigvals[operator]
        else:
            raise ValueError(f'Operator {operator} does not exist!')
        
    def basis(self):
        return self.__basis.T
    
    def eigenvalues_basis(self, operator):
        if operator in self.__eigvals_basis:
            return self.__eigvals_basis[operator]
        else:
            raise ValueError(f'Operator {operator} does not exist!')
        
    def sort_eigenvalues(self, operator, perm):
        if operator in self.__operators:
            self.__eigvals[operator] = self.__eigvals[operator][perm]
            self.__eigvecs[operator] = self.__eigvecs[operator].T[perm].T
        else:
            raise ValueError(f'Operator {operator} does not exist!')
        
    def find_basis(self, op1, op2, decimals=10):
        if op1 in self.__operators and op2 in self.__operators:
            # find degeneracies of operator 1
            _, groups = np.unique(np.round(self.__eigvals[op1], decimals=decimals), return_inverse=True)

            # find common basis of eigenvectors of both operators
            common_basis = np.empty(shape=(self.__num, 0))
            eval_op2 = np.empty(shape=(0))
            for g in np.unique(groups):
                evec_op1 = self.__eigvecs[op1][:, groups == g]

                # find eigenvalue in projected space
                eval, evec_proj = np.linalg.eig(np.conj(evec_op1).T @ self.__operators[op2] @ evec_op1)
                evec_common = evec_op1 @ evec_proj


                # add found eigenvectors to base
                common_basis = np.append(common_basis, evec_common, axis=1)
                eval_op2 = np.append(eval_op2, eval)
            # set member variable
            self.__basis = common_basis

            # set eigenvalues
            self.__eigvals_basis.clear()
            self.__eigvals_basis[op1] = self.__eigvals[op1]
            self.__eigvals_basis[op2] = eval_op2
            
        else:
            raise ValueError(f'Operator {op1} or {op2} does not exist!')

    def plot_state(self, state, which='m', ax=None):
        # get axes if none were specified
        if ax is None:
            ax = plt.gca()

        limit = max(abs(np.min(np.real(state))), abs(np.max(np.real(state))) )

        if 'real' == which or 'r' == which:
            limit = max(abs(np.min(np.real(state))), abs(np.max(np.real(state))))
            ax.imshow(np.real(state), cmap='twilight', label=f'$\\Re(\\Psi)$', vmin=-limit, vmax=limit)
        elif 'imag' == which or 'i' == which:
            limit = max(abs(np.min(np.imag(state))), abs(np.max(np.imag(state))))
            ax.imshow(np.imag(state), cmap='twilight', label=f'$\\Im(\\Psi)$', vmin=-limit, vmax=limit)
        elif 'mag' == which or 'm' == which:
            ax.imshow(np.abs(state), cmap='magma', label=f'$\\vert\\Psi\\vert^2$')

        # set labels
        ax.set_xlabel(f'$x$')
        ax.set_ylabel(f'$y$')

    def plot_potential(self, ax=None):
        # get axes if none were specified
        if ax is None:
            ax = plt.gca()
        
        # plot potential
        ax.imshow(self.__potential, cmap='magma')

        # add custom tick marks
        #ax.set_xticks(np.arange(self.__potential.shape[0]), np.linspace(-self.__width/2, self.__width/2, self.__num))
        #ax.set_yticks(np.arange(self.__potential.shape[0]), np.linspace(-self.__width/2, self.__width/2, self.__num))

        # set labels
        ax.set_xlabel(f'$x$')
        ax.set_ylabel(f'$y$')

    def plot_eigenvector(self, operator, num, which='m', ax=None):
        self.plot_state(np.moveaxis(self.__eigvecs[operator].reshape(self.__num, self.__num, -1), -1, 0)[num], which=which, ax=ax)

    def plot_eigenvalues(self, operator, eigval_range=None, ax=None):
        if operator in self.__operators:
            # get axes if none were specified
            if ax is None:
                ax = plt.gca()

            # calculate mask for range of eigenvalues
            if eigval_range is None:
                mask = np.full(self.__num**2, True)
            else:
                mask = ((np.arange(self.__num**2) >= eigval_range[0]) & (np.arange(self.__num**2) <= eigval_range[1]))

            # get x and y axis
            x = np.arange(self.__num**2)[mask]
            y = self.__eigvals[operator][mask]

            # plot
            ax.plot(x, y)

            # set labels
            ax.set_ylabel(f'EVs of ${operator}$')
        else:
            raise ValueError(f'Basis operator {operator} does not exist!')
        
    def normalize(self, state):
        state /= np.linalg.norm(state)
        return state

