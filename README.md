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

##### 2.1.1 Signature:

``apply_to(coefficients, input_basis='real', output_basis='real')``

##### 2.1.2 Parameters:

- coefficients (np.ndarray): An array of coefficients in the input space.
- input_basis (str): Specifies the space of the input coefficients. Valid values are 'real' and 'spectral'.
- output_basis (str): Specifies the space of the output coefficients. Valid values are 'real' and 'spectral'.

##### 2.1.3 Returns:

- np.ndarray: An array of coefficients in the output space.

##### 2.1.4 Raises:

- ValueError: If input_basis or output_basis is not 'real' or 'spectral'.

#### ``lattice``

Generates a discretized set of points representing the lattice structure in either the time or frequency domain.

##### Signature:

`lattice`

##### Parameters:

- output_basis (str): Specifies the space for which the lattice is generated. Valid values are:
- 'time': Generates a lattice for the time domain.
- 'frequency': Generates a lattice for the frequency domain.

##### Returns:

- np.ndarray: An array of lattice points in the specified output space.

##### Raises:

- ValueError: If output_basis is not 'real' or 'spectral'.

Time Lattice

The time lattice is created using a linear space from 0 to β, divided into n_time points. This discretization allows representing time on a grid for numerical calculations.

apply

Applies the Dirac operator to a set of spectral coefficients.

Signature:

Parameters:

    •	coefficients (np.ndarray): An array of spectral coefficients.

Returns:

    •	np.ndarray: An array of transformed spectral coefficients.

Raises:

    •	ValueError: If coefficients is not a 1-dimensional array.

eigenvalues

Computes the eigenvalues of the Dirac operator.

Signature:

Returns:

    •	np.ndarray: An array of eigenvalues.

eigenfunctions

Computes the eigenfunctions of the Dirac operator.

Signature:

Returns:

    •	list: A list of eigenfunctions.

to_frequency

Transforms time-domain values to frequency-domain values.

Signature:

Parameters:

    •	time_values (np.ndarray): An array of values in the time domain.

Returns:

    •	np.ndarray: An array of values in the frequency domain.

Raises:

    •	ValueError: If time_values is not a 1-dimensional array.

to_time

Transforms frequency-domain values to time-domain values.

Signature:

Parameters:

    •	frequency_values (np.ndarray): An array of values in the frequency domain.

Returns:

    •	np.ndarray: An array of values in the time domain.

Raises:

    •	ValueError: If frequency_values is not a 1-dimensional array.

Usage

To use the DiracOperator class, create an instance and call the desired methods with appropriate parameters. For example:
