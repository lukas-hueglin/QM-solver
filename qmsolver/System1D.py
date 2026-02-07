import scipy
import numpy as np
import matplotlib.pyplot as plt

from qmsolver import constants

class System1D:
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

        # create all operators
        self.x = np.linspace(-self.__width/2, self.__width/2, self.__num)
        self.__operators['x'] = np.diag(self.x)

        dx_row = np.zeros(self.__num)
        dx_row[-1] = 1 / (2*self.__step); dx_row[1] = -1 / (2*self.__step)
        self.__operators['dx'] = scipy.linalg.circulant(dx_row)

        dxx_row = np.zeros(self.__num)
        dxx_row[[-1, 1]] = 1 / (self.__step**2); dxx_row[0] = -2 / (self.__step**2)
        self.__operators['dxx'] = scipy.linalg.circulant(dxx_row)

        self.__operators['p'] = -1.0j*constants.hbar * self.op('dx')

        self.__operators['H'] = -constants.hbar**2 * self.op('dxx') / (2*constants.me) + np.diag(self.__potential)

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
            return self.__eigvecs[operator].T
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

    def plot_state(self, state, basis, eigval_range=None, which=('m'), plot_type='line', ax=None):
        if basis in self.__operators:
            # get axes if none were specified
            if ax is None:
                ax = plt.gca()

            # calculate mask for range of eigenvalues
            if eigval_range is None:
                mask = np.full(self.__num, True)
            else:
                mask = (self.__eigvals[basis] >= eigval_range[0]) & (self.__eigvals[basis] <= eigval_range[1])

            # project into basis of 'basis' operator eigenvectors
            x = self.__eigvals[basis][mask]
            y = self.__eigvecs[basis].T[mask] @ state

            if plot_type == 'line':
                if 'real' in which or 'r' in which:
                    ax.plot(x, np.real(y), color='red', label=f'$\\Re(\\Psi)$')
                if 'imag' in which or 'i' in which:
                    ax.plot(x, np.imag(y), color='royalblue', label=f'$\\Im(\\Psi)$')
                if 'mag' in which or 'm' in which:
                    ax.plot(x, np.square(np.abs((y))), color='black', label=f'$\\vert\\Psi\\vert^2$')
            elif plot_type == 'bar':
                width=10*(np.max(x) - np.min(x))/np.size(x)
                if 'real' in which or 'r' in which:
                    ax.bar(x, np.real(y), width=width, color='red', label=f'$\\Re(\\Psi)$')
                if 'imag' in which or 'i' in which:
                    ax.bar(x, np.imag(y), width=width, color='royalblue', label=f'$\\Im(\\Psi)$')
                if 'mag' in which or 'm' in which:
                    ax.bar(x, np.square(np.abs((y))), width=width, color='black', label=f'$\\vert\\Psi\\vert^2$')

            # set labels
            ax.set_xlabel(f'${basis}$')
            ax.legend()
        else:
            raise ValueError(f'Basis operator {basis} does not exist!')
        
    def plot_potential(self, ax=None):
        # get axes if none were specified
        if ax is None:
            ax = plt.gca()

        # generate test points
        x = self.__eigvals['x']
        y = self.__potential

        ax.plot(x, y, c='orange', ls='dashed')
        ax.fill_between(x, y, np.min(x), color='orange', alpha=0.3)

    def plot_eigenvector(self, basis, operator, num, which=('m'), ax=None):
        self.plot_state(self.__eigvecs[operator][:, num].T, basis, which=which, ax=ax)

    def plot_eigenvalues(self, operator, eigval_range=None, ax=None):
        if operator in self.__operators:
            # get axes if none were specified
            if ax is None:
                ax = plt.gca()

            # calculate mask for range of eigenvalues
            if eigval_range is None:
                mask = np.full(self.__num, True)
            else:
                mask = ((np.arange(self.__num) >= eigval_range[0]) & (np.arange(self.__num) <= eigval_range[1]))

            # get x and y axis
            x = np.arange(self.__num)[mask]
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

