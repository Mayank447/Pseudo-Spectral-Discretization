from scipy import special
import matplotlib.pyplot as plt
import numpy as np

class MF:
    def __init__(self, B, L):
        self.B = B
        self.L = L

    def phi_0_p_k(self, p, k):
        """
        p: non-negative integer < nu
        k: int
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
        """
        return lambda x, y: (
            np.sum([self.phi_0_p_k(p, k)(x,y) for k in range(-10,11)], axis=0)
        )
    
    def phi_n_p_k(self, n, p, k):
        alpha_p = 2 * np.pi * p/self.L
        sqrt_B = np.sqrt(self.B)
        return lambda x,y: (
            special.hermite(n)(sqrt_B*y + ((alpha_p + k*self.B*self.L)/sqrt_B)) * 
            self.phi_0_p_k(p,k)(x,y)
        )
    #This special.hermite step can be calculated onlt along a row and then extruded
    
    def phi_n_p(self, n, p):
        """
        n: non-negative integer (nth eigenvector)
        p: non-negative integer < nu
        """
        normalization = np.pow(-1,n%2) * 1/(np.sqrt(special.factorial(n) * np.pow(2,n)))
        return lambda x, y: (
            normalization *
            np.sum([self.phi_n_p_k(n, p, k)(x,y) for k in range(-10,11)], axis=0)
        )
    
nu = 4
L = 1
N = 1000

if __name__ == '__main__':
    temp = MF(2*np.pi * nu, L)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    z = temp.phi_n_p(3, 2)(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, z.real)
    print(L/(N**2) * np.sum(z * z.conjugate()))
    plt.show()
