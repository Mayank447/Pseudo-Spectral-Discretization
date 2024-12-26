import numpy as np
from scipy import special

I2PI = 2j * np.pi
K = 100

class MagneticField:
    """
    Spectrum class to represent the spectrum (eigenvalue, eigenfunctions) of the Dirac operator in the presence of a magnetic field.
    """

    def __init__(self, Nt, nu, N, dimension=1+2):
        """
        Constructor of the Spectrum class.
        N: Number of energy levels
        nu: Number of flux quanta (zero mode degenracy)
        Nt: number of time slices
        dimension: Dimension of the space-time (1 for time + 2 for space by default)
        """
        self.initialize(Nt, nu, N, dimension)
        

    def initialize(self, Nt, nu, N, dimension):
        """
        Some other internal parameters:
        B: magnetic field
        L: spatial length (Lx=Ly i.e. a square space lattice)
        p: flux quanta (0 <= p < nu)
        flux: magnetic flux (Not used)

        beta: simulation time
        omega: temporal frequency
        """

        self.Nt = Nt
        self.nu = nu
        self.N = N
        self.gap = 1  #energy gap [Free parameter]
        
        self._dimension = dimension
        self._spinor_dimension = 2
        self.total_num_of_dof = Nt * nu * N * self._spinor_dimension

        self.B = 0.5 * np.square(self.gap/self.N) # As per disc. w/ Julian on dim analys
        self.L = np.sqrt(2 * np.pi * self.nu/self.B)
        self.flux = self.B * (self.L**2)
        self.beta = Nt * self.gap # Beta is kind of a free parameter to be honest

        self._eigenvalues = self.compute_eigenvalues()
        
        # For lattice discretization - to be checked
        self._n_x = self.N
        self._n_y = self.nu


    def lamda_value(self, n):
        """
        Function to return the eigenvalue of phi_n_k (lambda) for a given n.
        """
        return np.sqrt(2 * self.B * n)

    def compute_eigenvalues(self):
        """
        Returns a list of mu_n (both positive and negative) and increasing in n
        
        Some logic behind implementation:
            mu_n**2 = lambda_n**2 + omega**2
            Now lambda_n = sqrt(2*B*n), for each energy level n there is a degeneracy of 2n except ground
            The code should be self explanatory now, in the last line we consider both signs for mu_n
        """
        n = np.arange(1, self.N + 1)
        omega = np.arange(self.Nt) # I think should be symmetric about 0

        # There can be some optimization done wrt to storage and memory since a lot of values are just repeat
        eigval = np.sqrt(
                    (omega.reshape(-1,1))**2 
                        + np.repeat(2 * self.B * n, self.nu)
                    ).reshape(-1)

        eigval = np.repeat(eigval, self._spinor_dimension)
        eigval[1::2] = -eigval[1::2]
        return eigval
    

    def get_index(self, w, n, p, sign):
        """
        w,n,p,sign: 1D Numpy array
        sign: Must be 0 for positive and 1 for negative
        """
        index = np.zeros_like(w)
        index += w * int(self.total_num_of_dof/(self._spinor_dimension * self.Nt))

        index += self.nu * (n-1)
        index += p
        index *= self._spinor_dimension
        return index + sign
    

    def get_w_n_p_sign_from_index(self, index):
        """
        index: 1D Numpy array
        Returns: w,n,p,sign: Each 1D numpy array
        """
        if(not isinstance(index, np.ndarray)):
            index = np.array([index]) # Make this change in fermion2D as well

        space_dof = int(self.total_num_of_dof/self.Nt)
        w = index // space_dof
        index %= space_dof

        sign = index % self._spinor_dimension
        index = index // self._spinor_dimension

        n = index // self.nu
        p = index % self.nu

        return np.array([w, n+1, p, sign])


###################### Working Tested code ############
    def phi_0_p_k(self, p, k):
        """
        p: non-negative integer < nu
        k: int
        Returns: 2D array
        """
        normalization = np.pow((self.B/(np.pi * (self.L**2))), 0.25)
        alpha_p = 2 * np.pi * p/self.L
        
        return lambda x, y: (
            normalization * 
            np.exp(1j * (alpha_p + k * self.L * self.B) * x) *
            np.exp(-self.B/2 * np.pow(y + (alpha_p/self.B) + k*self.L, 2))
        )
    
    def phi_0_p(self, p):
        """
        p: non-negative integer < nu
        Returns: 2D array
        """
        return lambda x, y: (
            np.sum([self.phi_0_p_k(p, k)(x,y) for k in range(-10,11)], axis=0)
        )
    
    def phi_n_p_k(self, n, p, k):
        """
        Returns: 1D Array
        """
        alpha_p = 2 * np.pi * p/self.L
        sqrt_B = np.sqrt(self.B)
        return lambda x,y: (
            special.eval_hermite(n, sqrt_B*y + ((alpha_p + k*self.B*self.L)/sqrt_B)) * 
            self.phi_0_p_k(p,k)(x,y)
        ).flatten()
    #This special.hermite step can be calculated only along a row and then extruded

    def phi_n_p(self, n, p):
        """
        n: non-negative integer (nth energy level)
        p: non-negative integer < nu (degeneracy)
        """
        normalization = np.pow(-1,n%2) * 1/(np.sqrt(special.factorial(n) * np.pow(2,n)))
        return lambda x, y: (
            normalization *
            np.sum([self.phi_n_p_k(n, p, k)(x,y) for k in range(-10,11)], axis=0)
        )
    
    # Rename the below as eigenfunction and account for both signs of mu
    def phi_w_n_p(self, w, n, p, sign):
        """
        w: Something to do with time
        n: non-negative integer (nth energy level)
        p: non-negative integer < nu
        sign: 0 for positive and -1 for negative
        """
        index = self.get_index(w,n,p,sign)
        mu = self._eigenvalues[index]
        lambd = np.sqrt(2 * self.B * n)
        normalization = 1/(np.sqrt(2 * self.beta * mu * (mu-w)))
        
        return lambda t,x,y: (
            normalization *
            np.exp(1j * w * np.repeat(t, self._spinor_dimension)) *
            np.array([(mu-w) * self.phi_n_p(n,p)(x,y), 
                      lambd * self.phi_n_p(n-1,p)(x,y)]).flatten('F')
        )
###################### Working Tested code ############

    def eigenfunction(self, index):
        w, n, p, sign = self.get_w_n_p_sign_from_index(index)
        return self.phi_w_n_p(w,n,p,sign)

    def transform(self, coefficients, input_basis, output_basis):
        """
        Function to transform the coefficients from real to spectral basis or vice versa.
        """
        if input_basis == output_basis in ["real", "spectral"]:
            return coefficients
        
        elif input_basis == "real" and output_basis == "spectral":
            pass
        
        elif input_basis == "spectral" and output_basis == "real":
            pass
        
        else:
            raise ValueError("Invalid input_basis or output_basis.")


    def scalar_product(self, f, g, output_basis="real"):
        if output_basis == "real":
            normalization_x_y = self.L**2/(self._n_x * self._n_y)
            return self.gap * normalization_x_y * (g @ f.transpose().conjugate())

        elif output_basis == "spectral":
            return g @ f.transpose().conjugate()
        
        else:
            raise ValueError(f"Invalid output_basis {output_basis}.")


    def lattice(self, output_basis="real"):
        """
        Function to return the lattice of the spectrum.
        """
        if output_basis == "real":
            t = np.arange(self.Nt)
            x = np.linspace(0, self.L, self._n_x, endpoint=False)
            y = np.linspace(0, self.L, self._n_y, endpoint=False)
            t, x, y = np.meshgrid(t, x, y, indexing="ij")
            return t.flatten(), x.flatten(), y.flatten()

        elif output_basis == "spectral":
            return self.eigenvalues

        else:
            raise ValueError(f"Invalid output_basis {output_basis}.")


    @property
    def dimension(self):
        return self._dimension


if __name__ == "__main__":
    spectrum = MagneticField(3, 3, 10)
    sample_points = spectrum.lattice()
    # e = spectrum.eigenfunction(0)(*sample_points)
    # f = spectrum.eigenfunction(2)(*sample_points)
    # print(spectrum.scalar_product(e,f))
    # print(e)

    eigenfunctions = spectrum.eigenfunction(np.arange(spectrum.total_num_of_dof))(
        *sample_points
    ).reshape(spectrum.total_num_of_dof, -1)

    # assert np.allclose(
    #     spectrum.scalar_product(eigenfunctions, eigenfunctions),
    #     np.eye(*eigenfunctions.shape),
    # )

## Couple of Notes:
## 1. n > 0 as chi_0 is not defined
## 2. A lot of optimizations can be done