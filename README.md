# Pseudo-Spectral-Discretization

# DiracOperator Class

The `DiracOperator` class implements the Dirac operator with methods for applying it to spectral coefficients, computing eigenvalues and eigenfunctions, generating a lattice structure, and transforming values between time and frequency spaces.

## `lattice` Method

### Description

The `lattice` method generates a discretized set of points representing the lattice structure in either the time or frequency domain. This is essential for numerical simulations, where continuous variables are approximated by discrete points.

### Method Signature

```python
def lattice(self, output_space):
    """
    Return the lattice of the Dirac operator as per the given output space.

    Parameters:
    output_space (str): The space for which to generate the lattice. 
                        Must be either 'time' or 'frequency'.

    Returns:
    np.ndarray: The lattice points in the specified output space.

    Raises:
    ValueError: If the output space is not 'time' or 'frequency'.
    """
```    

### Parameters

- output_space (str): Specifies the space for which the lattice is generated. Valid values are:
- 'time': Generates a lattice for the time domain.
- 'frequency': Generates a lattice for the frequency domain.

### Returns

np.ndarray: An array of lattice points in the specified output space.

### Raises

ValueError: If output_space is not 'time' or 'frequency'.

### Time Lattice

The time lattice is created using a linear space from 0 to $\beta$, divided into n_time points. This discretization allows representing time on a grid for numerical calculations.