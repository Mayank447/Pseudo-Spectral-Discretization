# Pseudo-Spectral-Discretization

# DiracOperator Class

The `DiracOperator` class implements the Dirac operator with methods for applying it to vectors in real-space and spectral representation. It takes a spectrum as defined below.

- *pseudospectral* folder contains all the source code of the project which are mainly `.py` files broadly categorized into DiracOperator and Spectra python files.
- *pseudospectral/spectra* folder contains spectrum class implementations.
- *example_scripts* folder contains example scripts to demonstrate the usage of the `DiracOperator` and `Spectrum` classes.
- *tests* folder contains all the pytests written for the project.

<pre>
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
</pre>

#### Setup

- git clone the repository - `git clone https://github.com/Mayank447/Pseudo-Spectral-Discretization.git`
- cd into the repository - `cd Pseudo-Spectral-Discretization`
- run the command `pip install .`

#### Usage

- After setting up the package, you can import the classes and methods from `pseudospectral` module in your code for example as follows:
  ``from pseudospectral import DiracOperator, FreeFermions2D``
- You may further refer to:
  - `example_scripts` folder for example script usage of the DiracOperator, Spectrum classes, methods and a boiler-plate to begin with.