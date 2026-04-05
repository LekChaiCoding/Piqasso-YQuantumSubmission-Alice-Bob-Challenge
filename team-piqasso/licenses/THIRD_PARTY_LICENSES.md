## Third-Party Licenses Used by `team-piqasso`

This inventory is based on the libraries directly referenced in the `team-piqasso` notebooks and Python scripts in this repository snapshot.

### Upstream challenge material

| Component | How it is used here | License status in this checkout |
| --- | --- | --- |
| Alice & Bob challenge notebook and bundled challenge materials under `challenge/` | Used as the starting point and reference implementation for the submission work | No separate upstream license file was found in this local checkout. Attribution is preserved in `ACKNOWLEDGMENTS.md`, but no license is asserted here without an upstream file to cite. |

### Direct third-party libraries referenced in code

| Component | Where it appears | License |
| --- | --- | --- |
| Python and the Python standard library | Runtime for notebooks and scripts | PSF-2.0 |
| `dynamiqs` | Main quantum-system simulation backend in notebooks and scripts | Apache-2.0 |
| `jax` / `jax.numpy` | JIT, autodiff, and array operations | Apache-2.0 |
| `numpy` | Numerical arrays and utility functions | BSD-3-Clause |
| `scipy` | Optimization, fitting, linear algebra, integration, and statistics | BSD-3-Clause |
| `matplotlib` | Plotting and figure export | Matplotlib license (BSD-style) |
| `cmaes` | SepCMA optimizer implementation | MIT |
| `torch` / PyTorch | PPO and neural-network-based optimizer experiments | BSD-3-Clause |
| `scikit-learn` | Gaussian-process components in pi-pulse tooling | BSD-3-Clause |
| `IPython` | Notebook display helpers in Colab-style scripts | BSD-3-Clause |
| `qutip` | Cat-size proof script used for related figure generation | BSD-3-Clause |
| `cma` | Alternate CMA-ES implementation used in one experimental script | BSD-3-Clause |

### Scope note

This file is intentionally limited to components that are directly imported by the checked-in `team-piqasso` code or clearly referenced by the surrounding project documentation.

If new libraries are added later, this inventory should be updated at the same time.
