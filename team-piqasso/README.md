# Team Piqasso Submission Notes

This folder holds the team’s exploratory notebooks, scripts, figures, and PPO tooling for the Alice & Bob cat-qubit challenge.

## Folder layout

```
team-piqasso/
├── README.md                          ← this file
├── PiqassoNotebook.ipynb              ← main submission narrative (CMA-ES, PPO, cat-size proof)
├── four-parameter-optimizer.ipynb     ← CMA-ES-style loop over Re/Im(g₂), Re/Im(ε_d)
├── rl_refinement_notebook.ipynb       ← PPO refinement over the same four controls
├── ppo_batched_parallel_search.py     ← modular PPO runner (surrogate or Lindblad backend, plots)
├── Graphs/                            ← figures and PPO run exports
│   ├── README.md
│   ├── requirements.txt               ← extra graph-related deps (if used)
│   ├── cat_size_proof.png             ← from repo-root `cat_size_proof.py` (QuTiP proof)
│   ├── optimizer_progress.png, optimized_Tx_Tz_combined.png, …
│   ├── ppo_results/                   ← PPO training curves and decay snapshots
│   └── Alex/outputs/ppo_results/      ← another PPO results export set
├── Notebooks/                         ← extra topic notebooks
│   ├── AddZ.ipynb
│   ├── Detuning.ipynb
│   ├── Drift.ipynb
│   └── notebook.ipynb
├── outputs/ppo_results/               ← default-style PPO plot output folder (when used)
└── Scripts/                           ← Python utilities and experiments
    ├── cmas.py                        ← CMA-ES / optimization driver used with notebooks
    ├── ppo_batched_parallel_search.py ← copy/variant of root PPO script (prefer root file)
    ├── Tx_Tz_optimization.py
    ├── optimize_cat.py, optimize_cat_fast.py, optimize_cat_colab.py, optimize_cat_old.py
    ├── pi_pulse_pipeline.py, pi_pulse_pipeline_old.py
    ├── alpha_optimizer.py, parameter_analysis.py, leon.py, Piqasso.py
    └── …
```

The repository root also includes **`cat_size_proof.py`** (not inside `team-piqasso/`): QuTiP steady-state / Wigner check that **ε_d / g₂** sets cat size; figures often live under **`team-piqasso/Graphs/`**.

## What the main pieces do

### Notebooks (top level)

- **`four-parameter-optimizer.ipynb`** — Optimizes the four real controls behind **g₂** and **ε_d** using **`dynamiqs`** simulations, estimates **T_X**, **T_Z**, and bias **η = T_Z / T_X**, and uses a CMA-ES-style search (see **`Scripts/cmas.py`**) to improve a lifetime / bias objective.
- **`rl_refinement_notebook.ipynb`** — PPO-style refinement on the same physical controls, Gaussian policy, clipped updates, replay, aligned with the challenge Hamiltonian when using the Lindblad path.
- **`PiqassoNotebook.ipynb`** — Write-up for the submission: guiding question, CMA-ES vs PPO comparison, link to **`cat_size_proof`**, and figures under **`Graphs/`**.

### `ppo_batched_parallel_search.py` (this folder)

- Same four-parameter search space, **physics-informed seed**, **`SimulatorBackend`** protocol:
  - **`SurrogateBackend`** — fast analytic metrics.
  - **`LindbladBackend`** — full storage–buffer **`dynamiqs.mesolve`**, exponential fits to logical observables.
- Reward (JIT): **0.5 log(T_X T_Z) − λ_η (log η − log η_target)²**, with a **hard** gate on **η** outside **[η_min, η_max]** by default.
- **Replay buffer** mixes past transitions with the fresh batch for PPO updates.
- Writes training and decay figures under **`PPOConfig.output_dir`** (often **`team-piqasso/outputs/...`** or a path you pass with **`--output-dir`**). Example bundles also appear under **`Graphs/ppo_results/`** and **`Graphs/Alex/outputs/ppo_results/`**.

### `Scripts/`

- **`cmas.py`** — CMA-ES and related optimization / plotting used from notebooks.
- **`optimize_cat*.py`**, **`pi_pulse_pipeline*.py`**, **`Tx_Tz_optimization.py`**, **`alpha_optimizer.py`**, **`parameter_analysis.py`**, **`leon.py`**, **`Piqasso.py`** — Additional experiments and pipelines; names with **`_old`** / **`_colab`** are alternate or legacy versions.
- **`Scripts/ppo_batched_parallel_search.py`** — Treat as a secondary copy; for a single source of truth, run **`team-piqasso/ppo_batched_parallel_search.py`** from the repo root.

### `Notebooks/`

- **`Drift.ipynb`**, **`Detuning.ipynb`**, **`AddZ.ipynb`**, **`notebook.ipynb`** — Topic-specific explorations (drift, detuning, Z-type additions, etc.).

### `Graphs/`

- Static figures for the write-up and optimizer comparisons; **`Graphs/README.md`** describes graph-specific notes if present.
- PPO epoch curves and decay snapshots may be duplicated under **`Graphs/ppo_results/`** and **`Graphs/Alex/outputs/ppo_results/`** from different runs.

## How to run

1. From the **repository root**, install dependencies:  
   `pip install -r requirements.txt`  
   (If you use QuTiP for **`cat_size_proof.py`**, install **qutip** separately if it is not listed.)

2. **CMA / four-parameter notebook**  
   Open **`team-piqasso/four-parameter-optimizer.ipynb`** and run top to bottom.

3. **RL / PPO notebook**  
   Open **`team-piqasso/rl_refinement_notebook.ipynb`** and run top to bottom (GPU optional if JAX + CUDA are set up).

4. **Batched PPO script** (from repo root):

   ```bash
   python team-piqasso/ppo_batched_parallel_search.py
   ```

   GPU example (adjust paths and venv to your machine):

   ```bash
   python team-piqasso/ppo_batched_parallel_search.py --backend lindblad --epochs 1000 --batch-size 12 --replay-sample-size 64 --snapshot-every 100
   ```

   Useful flags: **`--log-every`**, **`--log-candidates`**, **`--quiet`**, **`--eta-min`**, **`--eta-max`**, **`--target-bias`**, **`--lambda-eta`**, **`--eval-x-tfinal`**, **`--eval-z-tfinal`**, **`--eval-nsave`**, **`--output-dir`**.

   Quick surrogate smoke test:

   ```bash
   python team-piqasso/ppo_batched_parallel_search.py --backend surrogate --epochs 2 --batch-size 4 --replay-sample-size 4 --ppo-epochs 1 --snapshot-every 1 --output-dir team-piqasso/outputs/ppo_batched_parallel_smoke
   ```

   Reduced Lindblad smoke test:

   ```bash
   python team-piqasso/ppo_batched_parallel_search.py --backend lindblad --epochs 1 --batch-size 1 --replay-sample-size 1 --ppo-epochs 1 --snapshot-every 1 --na 8 --nb 3 --nsave 16 --x-tfinal 0.8 --z-tfinal 6.0 --output-dir team-piqasso/outputs/ppo_batched_parallel_lindblad_smoke
   ```

5. **Cat size proof (repo root)**  
   `python cat_size_proof.py`  
   Point the save path inside the script to e.g. **`team-piqasso/Graphs/cat_size_proof.png`** if your environment does not use the default output paths.

If JAX reports **`CudaDevice`**, the heavy simulations can use the GPU; **`CpuDevice`** means CPU-only (still correct, slower).

## Purpose (plain language)

The notebooks and **`ppo_batched_parallel_search.py`** are meant to be easy to read and extend: they search over **g₂** and **ε_d** in the same spirit as the challenge notebook, score candidates with **T_X**, **T_Z**, and **η**, and save plots so you can see both **whether** the optimizer improved and **how** it explored parameter space.
