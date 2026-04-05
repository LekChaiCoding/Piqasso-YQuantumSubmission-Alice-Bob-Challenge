## Acknowledgments

### Alice & Bob materials

This project builds on the Alice & Bob cat-qubit challenge materials included in this repository, especially the provided challenge notebook and the modeling approach used there.

We specifically acknowledge:

- Alice & Bob's challenge notebook under `challenge/`.
- Alice & Bob's `dynamiqs` quantum simulation package, which is used throughout the `team-piqasso` notebooks and scripts.
- The overall cat-qubit stabilization setup, notation, and benchmarking context provided in the challenge materials.

### What was adapted in `team-piqasso`

Within `team-piqasso`, the team created its own notebooks, scripts, plots, optimizer variants, and analysis on top of those upstream materials. That includes:

- CMA-ES-based tuning and reward experiments.
- PPO-based optimization variants and batched search tooling.
- Drift, detuning, and parameter-analysis notebooks.
- Figures and exports generated from those experiments.

### Important attribution note

No standalone upstream license file for the bundled challenge materials was found in this local repository snapshot. Because of that, this notice does two things:

- It preserves explicit attribution to Alice & Bob for the provided challenge code and materials.
- It does not invent a license for those bundled challenge files where one was not present in the checkout.

If this project is redistributed beyond the challenge submission context, the upstream Alice & Bob license for the bundled challenge materials should be verified directly from the original source before republishing those files.
