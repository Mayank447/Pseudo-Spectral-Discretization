# Pseudo-Spectral-Discretization

This codebase contains the implementation of the DiracOperator class, which is used for applying the Dirac operator to spectral coefficients, computing eigenvalues and eigenfunctions, generating lattice structures, and transforming values between real and spectral spaces.

### Directory Structure

- `pseudospectral` folder contains all the source code of the project which are mainly `.py` files broadly categorized into Dirac Operator and Spectra python files.
- `pseudospectral/spectra` folder contains spectrum class implementations.
- `example_scripts` folder contains example scripts to demonstrate the usage of the DiracOperator and Spectrum classes.
- `tests` folder contains all the pytests written for the project.

`<pre>`
.
├── pseudospectral
    ├── __init__.py
    ├── dirac_operator.py
    ├── spectra
        ├── derivative_1D.py
        ├── free_fermions_2D.py
├── pyproject.toml
├── README.md
├── tutorial.md
├── example_scripts
    ├── cosine_derivative.py
├── LICENSE
└── tests
    ├── conftest.py
    ├── test_real_basis.py
    ├── test_spectral_basis.py
    ├── test_transform.py
    ├── test_spectra.py
`</pre>`

#### Setup

- git clone the repository - `git clone https://github.com/Mayank447/Pseudo-Spectral-Discretization.git`
- cd into the repository - `cd Pseudo-Spectral-Discretization`
- run `pip install .`

#### Usage

- After setting up the package, you can import the classes and methods from `pseudospectral` module in your code as follows:
  ``from pseudospectral import DiracOperator, FreeFermions2D``
- You may further refer to:
  - `example_scripts` folder for example script usage of the DiracOperator, Spectrum classes, methods and a boiler-plate to begin with.
  - `tutorial.md` for a step-by-step guide on how to use the classes and methods and a bit of theory behind the implementation (history behind the project).

## DiracOperator Class

The DiracOperator class provides several methods to work with the Dirac operator in both real and spectral domains.

### 1 Initialization ``DiracOperator(spectrum)``

#### 1.1 Args:

- spectrum: The spectrum class object that represents the spectrum of the given system / operator.

### 2 Methods

#### 2.1 ``apply_to``

Applies the given Dirac operator to the input coefficients as per the given input and output spaces.

##### 2.1.1 Signature: ``apply_to(coefficients, input_basis='real', output_basis='real')``

##### 2.1.2 Parameters:

- coefficients (np.ndarray): An array of coefficients in the input space.
- input_basis (str): Specifies the space of the input coefficients. Valid values are 'real' and 'spectral'.
- output_basis (str): Specifies the space of the output coefficients. Valid values are 'real' and 'spectral'.

##### 2.1.3 Returns:

- np.ndarray: An array of coefficients in the output space.

##### 2.1.4 Raises:

- ValueError: If input_basis or output_basis is not 'real' or 'spectral'.

## Derivative1D Class

The Derivative1D class provides methods to work with the 1D derivative operator in both real and spectral domains with periodic boundary conditions.

### 1 Initialization ``Derivative1D``

#### 1.1 Args:

- L: The length of the domain.
- num_points: The number of points in the discretization.

### 2 Attributes:

- L: The length of the domain.
- num_points: The number of points in the discretization.
- eigenvalues: Eigenvalues of the 1D derivative operator.

### 3 Methods

#### 3.1 ``eigenfunctions``

Returns the eigenfunction of the 1D derivative operator at the given eigenvalue index i.e. exp(ikx).

##### 3.1.1 Signature: `eigenfunctions(eigenvalue_index)`

##### 3.1.2 Parameters:

- eigenvalue_index (int): The index of the eigenvalue. This range is from 0 to L - 1.

##### 3.1.3 Returns:

- lambda function: The eigenfunction of the 1D derivative operator which can be sampled at the given points by passing the array as an argument.

##### 3.1.4 Raises:

- ValueError: If eigenvalue_index is not in the range from 0 to L - 1.

#### 3.2 ``_eigenvalues``

Private function to compute the eigenvalues of the 1D derivative operator.

#### 3.3 ``transform``

Transforms the input coefficients from the real space to the spectral space and vice-verse.

##### 3.3.1 Signature: `transform(coefficients, input_basis='real', output_basis='real')`

##### 3.3.2 Parameters:

- coefficients (np.ndarray): An array of coefficients in the input space.
- input_basis (str): Specifies the space of the input coefficients. Valid values are 'real' and 'spectral'.
- output_basis (str): Specifies the space of the output coefficients. Valid values are 'real' and 'spectral'.

##### 3.3.3 Returns:

- np.ndarray: An array of coefficients in the output space.

##### 3.3.4 Raises:

- ValueError: If input_basis or output_basis is not 'real' or 'spectral'.

#### 3.4 ``lattice``

Generates a discretized set of points representing the lattice structure in either the time or frequency domain.

##### 3.4.1 Signature: `lattice(output_basis)`

##### 3.4.2 Parameters:

- output_basis (str): Specifies the space for which the lattice is generated. Valid values are:
  - 'real': Generates a lattice for the time domain.
  - 'spectral': Generates a lattice for the frequency domain.

##### 3.4.3 Returns:

- np.ndarray: An array of lattice points in the specified output space.

##### 3.4.4 Raises:

- ValueError: If output_basis is not 'real' or 'spectral'.
