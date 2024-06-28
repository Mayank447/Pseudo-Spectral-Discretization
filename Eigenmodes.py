import numpy as np
from scipy.linalg import eigh

# from scipy.special import hermite (Useful later)
import matplotlib.pyplot as plt

# Constants
L = 1.0  # Side length of the square
B = 2 * np.pi / L**2  # Magnetic field strength


class DiracOperator:
    def __init__(self, N: int):
        self.N = N
        self.dx = L / N
        # self.A = 1j * (np.kron(self.Dy, np.eye(N)) +
        #                B * np.kron(np.eye(N), self.Dy))
        # self.A_dag = np.conjugate(self.A.T)
        # self.eigenvalues, self.eigenvectors = self.find_eigenmodes()
        # self.dirac_modes = self.construct_dirac_eigenmodes()
        # self.gamma0, self.gamma1, self.gamma2 = self.get_dirac_matrices()

    def get_dirac_matrices(self):
        """ """
        gamma0 = np.array([[0, 1], [1, 0]], dtype=complex)
        gamma1 = np.array([[0, 1], [-1, 0]], dtype=complex)
        gamma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return gamma0, gamma1, gamma2

    def dirac_operator(self, L, B, gamma0, gamma1, gamma2):
        """
        Constructs the Dirac operator matrix.
        L: side length of the square
        B: magnetic field strength
        N: number of discretization points
        gamma1, gamma2: Dirac matrices
        """

        N = self.N
        dx = L / N
        # dy = L / N (Useful later)

        # Create derivative operators with periodic boundary conditions
        D = (
            np.diag(-2 * np.ones(N))
            + np.diag(np.ones(N - 1), 1)
            + np.diag(np.ones(N - 1), -1)
        )
        D[0, -1] = D[-1, 0] = 1
        D /= dx**2

        # Identity matrices
        I_N = np.eye(N)
        # I_2 = np.eye(2)

        # Constructing the Dirac operator
        D_x = np.kron(I_N, D)  # Derivative in the x-direction
        D_y = np.kron(D, I_N)  # Derivative in the y-direction

        # Construct the full operator
        A = (
            np.kron(gamma1, D_x)
            + np.kron(gamma2, D_y)
            + np.kron(gamma0, B * np.kron(I_N, I_N))
        )

        return A

    # Code up the apply_to function

    def find_eigenmodes(self, squared_operator):
        """
        Finds the eigenvalues and eigenmodes of the squared operator.
        """
        # print(squared_operator)
        eigenvalues, eigenvectors = eigh(squared_operator)
        # eigh solves eigenvalues for a Hermitian, real symmetric matrix
        return eigenvalues, eigenvectors

    def construct_dirac_eigenmodes(self, eigenvalues, eigenvectors):
        """
        Constructs the eigenmodes of the original Dirac operator
        from the eigenmodes of the squared operator.
        """
        dirac_modes = []
        for i, lambda_sq in enumerate(eigenvalues):
            if lambda_sq > 0:
                lambda_val = np.sqrt(lambda_sq)
                phi_n = eigenvectors[:, i]
                dirac_modes.append((lambda_val, phi_n))
        return dirac_modes


def main():
    N = 10  # Number of discretization points
    D = DiracOperator(N)
    gamma0, gamma1, gamma2 = D.get_dirac_matrices()
    dirac_op = D.dirac_operator(L, B, gamma0, gamma1, gamma2)
    eigenvalues, eigenvectors = D.find_eigenmodes(dirac_op)
    dirac_modes = D.construct_dirac_eigenmodes(eigenvalues, eigenvectors)

    # Print and plot eigenmodes
    for i, (lambda_val, phi_n) in enumerate(dirac_modes[:5]):
        print(f"Eigenmode {i}: Eigenvalue = {lambda_val}")
        plt.plot(np.abs(phi_n) ** 2, label=f"Mode {i}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
