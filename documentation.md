# Tutorial 

## Dirac Operator - 
This tutorial covers the application of a finite-dimensional linear operator using its spectral representation.

## Usage -

The linear operator is defined by its spectral representation which is given as a `Spectrum` object.

### Defining the `Spectrum`

Defining a spectral representation (a class) [input - eigenfunctions, corresponding eigenvalues, space]. User gives the spectrum object to the Dirac Operator while initializing the object. [We will be using a linear function representation in place of the matrix representation]
...

```
# Example 1D derivative:
class Derivative1D:
    def __init__(self, num_lattice_points):
        self.num_lattice_point = num_lattice_points
        self.L = 1
        self.eigenvalues = self._eigenvalue(..)

    def transform(self, input_vector, input_space, output_space):
        if input_space == 'real' and output_space == 'real':
            return input_vector

        # Perform the discrete Fast Fourier transform to go from real to spectral space
        elif input_space == 'real' and output_space == 'spectral':
            return np.fft.fft(input_vector) / np.sqrt(self.n_real)
        
        elif input_space == 'spectral' and output_space == 'spectral':
            return input_vector
        
        elif input_space == 'spectral' and output_space == 'real':
            # Perform the inverse discrete Fast Fourier transform to go from spectral to real space
            return np.fft.ifft(input_vector) * np.sqrt(self.n_real)
        
        else:
            raise ValueError("Unsupported space transformation.")
```

### Using the operator

Using the operator is straightforward afterwards.
Take your `spectral_representation` from the last section and define your input:
```
# Example: Represented in real space.
# This happens to be an eigenvector:
input_vector = [...]
```
Then, you can call
```
operator = Operator(spectral_representation)
result = operator.apply_to(input_vector)

# because input_vector is an eigenvector:
assert result == eigenvalue * input_vector
```
to get the result.

Explain what the code does!

# Testing -
assert expected == actual

# Development
Default values for input, output space to be real.
input_space='real', output_space='real'

```
def DiracOperator:
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def apply_to(input_vector):
        ...
        self.spectrum.transform()
```