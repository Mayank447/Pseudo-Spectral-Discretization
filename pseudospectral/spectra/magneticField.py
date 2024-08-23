import numpy as np
from scipy import special

I2PI = 2j * np.pi

class MagneticField:
    """
    Spectrum class to represent the spectrum (eigenvalue, eigenfunctions) of the Dirac operator in the presence of a magnetic field.
    """

    def __init__(self, Nt, nu, N, dimension=2+1):
        """
        Constructor of the Spectrum class.
        Nt: number of time slices
        nu: Number of flux quanta (zero mode degenracy)
        N: Number of energy levels
        # flux: magnetic flux
        """

        self.gap = 1  #energy gap [Free parameter]
        self._dimension = dimension
        self._eigenvalues = None

        self.Nt = Nt
        self.nu = nu
        self.N = N

        self.n = np.arange(self.N)
        self.p = np.arange(self.nu)
        self.omega = np.arange(self.Nt)
        
        self.L = np.sqrt(4 * np.pi * self.N**2/self.gap**2)
        self.B = 0.5 * np.square(self.gap/self.N)
        self.flux = self.B * (self.L**2)
        self.beta = Nt * self.gap

        # For lattice discretization
        self._n_x = self.nu
        self._n_y = self.N


    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._eigenvalues = self.compute_eigenvalues()
        return self._eigenvalues
        
    def compute_eigenvalues(self):
        eigval = np.sqrt(
                    (self.omega.reshape(-1,1))**2 + np.repeat(2 * self.B * self.n, 2 * self.nu)[self.nu :]
                 ).reshape(-1)
        
        eigval[1::2] = -eigval[1::2]
        return eigval 
    
    def lamda_value(self, n):
        """
        Function to return the eigenvalue (lambda) for a given n.
        """
        return np.sqrt(2 * self.B * n)


    def hermite_operator(self, n):
        return lambda x: np.exp(-(x**2)/2) * special.hermite(n)(x)

    def phi(self, n, p):
        """
        n: non-negative integer
        p: non-negative integer < nu
        """
        normalization = (
            (-1**(n%2)) * np.pow(self.B/(np.pi * self.L**2), 0.25) / np.sqrt(np.pow(2, n) * special.factorial(n))
        )

        # x, y are numpy arrays whereas k is a non-negative integer
        return lambda x, y, k: (
            normalization * 
            np.sum(
                self.hermite_operator(n)(
                    np.sqrt(self.flux) * (y/self.L + np.arange(-k, k+1).reshape(-1,1) + p/self.nu)
                ) * 
                np.exp(I2PI * (p + self.nu * np.arange(-k, k+1)).reshape(-1,1) * x / self.L)
            , axis=0)
        )

    def eigenfunction(self, index):
        omega = self.omega[index // ((2* self.N - 1) * self.mu)]
        lamda = (index % ((2* self.N - 1) * self.mu) + self.mu)
        n = self.n[lamda // (2*self.mu)]
        p = self.p[lamda % (2*self.mu)]
        mu = np.sqrt(omega**2 + self.lamda_value(n)**2)

        lambda t, x, y: (
            np.exp(1j * omega * t) / np.sqrt(2 * self.beta * mu * (mu - omega)) *
            np.ravel([
                (mu - omega) * self.phi(n, p)(x, y, 100),
                self.lamda_value(n) * self.phi(n-1, p)(x, y, 100)
            ], 'F')
        )


    def transform(self, coefficients, input_basis, output_basis):
        """
        Function to transform the coefficients from real to spectral basis or vice versa.
        """
        if input_basis == output_basis in ["real", "spectral"]:
            return coefficients
        
        elif input_basis == "real" and output_basis == "spectral":
            return self.real_to_spectral(coefficients)
        
        elif input_basis == "spectral" and output_basis == "real":
            return self.spectral_to_real(coefficients)
        
        else:
            raise ValueError("Invalid input_basis or output_basis.")


    def inner_product(self, f, g):
        return g @ f.transpose().conjugate()

    def lattice(self, output_basis="real"):
        """
        Function to return the lattice of the spectrum.
        """
        if output_basis == "real":
            t = np.arange(self.Nt)
            x = np.linspace(0, self.L, self._n_x, endpoint=False)
            y = np.linspace(0, self.L, self._n_y, endpoint=False)
            t, x, y = np.meshgrid(t, x, y, indexing="ijk")
            return t.flatten(), x.flatten(), y.flatten()

        elif output_basis == "spectral":
            return self.eigenvalues

        else:
            raise ValueError(f"Invalid output_basis {output_basis}.")

    @property
    def dimension(self):
        return self._dimension


if __name__ == "__main__":
    M = MagneticField(3, 3, 10)
    print(M.eigenvalues)
    # v = M.phi(2, 1)(np.array([0,1,2]), np.array([1,1,1]), 100)
    # print(v)