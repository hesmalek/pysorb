# pysorb

`pysorb` is a Python library for adsorption modelling across equilibrium and dynamic kinetic workflows. It connects reusable numerical solvers for scripted scientific computing with GUI-facing adsorption workflows.

The current package supports pure-component isotherm fitting, multicomponent mixture prediction, fixed-bed breakthrough simulation, Crank diffusion uptake analysis, and zero length column (ZLC) modelling. The numerical routines use NumPy and SciPy, with Numba-compatible kernels used in the mixture and breakthrough solvers for repeated numerical evaluation.

The package contains five solver areas:

* `pysorb.fit` for adsorption isotherm fitting.
* `pysorb.mix` for multicomponent equilibrium calculations.
* `pysorb.bt` for fixed-bed breakthrough simulations.
* `pysorb.crank` for Crank diffusion uptake calculations.
* `pysorb.zlc` for zero length column simulations.

The current package release is `0.2.2`; version `0.1.0` was used as an initial PyPI test release.

## Citation

Citation metadata is provided in `CITATION.cff`.

Archived release DOI: https://doi.org/10.5281/zenodo.20480051

## Installation

```bash
pip install pysorb
```

For local development from this repository:

```bash
pip install -e .
```

## Examples and GUI

Supplementary examples are available in `Examples/`:

* `Examples/notebooks/` contains a basic Jupyter notebook for scripted use.
* `Examples/sessions/` contains saved GUI/session input examples.

The standalone Windows GUI executable is available from the GitHub release assets:

https://github.com/hesmalek/pysorb/releases/tag/v0.2.2

## Basic Usage

```python
from pysorb.fit import fit_isotherm
from pysorb.mix import parse, iast
from pysorb.bt import run_simulation
from pysorb.crank import crank_uptake
from pysorb.zlc import linmodel
```

### Isotherm fitting

```python
import numpy as np
from pysorb.fit import fit_isotherm

P = np.array([0.1, 0.3, 0.7, 1.2, 2.0])
T = np.full_like(P, 298.15)
q = 1.5 * 0.8 * P / (1 + 0.8 * P)

fit = fit_isotherm(P, q, T, "Langmuir", mode="single")

# parameters[1] = log10(b), so b = 10**parameters[1]
print(fit.parameters)
```

### Mixture equilibrium

```python
from pysorb.mix import parse, iast

components = [
    {"MoleculeName": "A", "isotherms": [["Langmuir", 1.0, 0.5]]},
    {"MoleculeName": "B", "isotherms": [["Langmuir", 2.0, 0.2]]},
]

parsed = parse(components)
result = iast([1.0, 0.5], parsed=parsed)
```

### Breakthrough

```python
from pysorb.bt import run_simulation

result = run_simulation(inputs)
```

### Crank diffusion

```python
from pysorb.crank import crank_uptake

q_over_qinf = crank_uptake([0, 1, 2], D_R2=0.001)
```

### ZLC

```python
from pysorb.zlc import linmodel

t, c, c_preload, q, q_preload, sigsum = linmodel(
    Lsat=20,
    DR2=6.4e-3,
    gamma=0.001,
    tsat=50,
    t_end=80,
    NN="auto",
)
```
