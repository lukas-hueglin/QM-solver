import scipy

# physical constants
e = 1 # [e] = e
h_bar = scipy.constants.hbar / scipy.constants.e * 1e9 # [h] = eV*ns
eps_0 = scipy.constants.epsilon_0 / scipy.constants.e * 1e-9  # [eps0] = e^2/(eV*nm)
m_e = scipy.constants.m_e / scipy.constants.e #[m_e] = eV*ns^2/nm^2