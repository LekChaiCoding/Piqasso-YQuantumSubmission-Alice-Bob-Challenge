# Team Piqasso Submission Notes

This folder contains standalone exploratory tooling:

- `four-parameter-optimizer.ipynb`
- `rl_refinement_notebook.ipynb`
- `ppo_cat_gpu_search.py`
- `ppo_batched_parallel_search.py`
- `parity_decay_optimizer.py`

## What it does

The notebook builds a simple optimization loop for the four real control knobs behind the complex cat-stabilization parameters:

- `Re(g_2)`
- `Im(g_2)`
- `Re(epsilon_d)`
- `Im(epsilon_d)`

It uses `dynamiqs` to simulate logical decay, estimates proxy lifetimes `T_X` and `T_Z`, and then applies CMA-ES to improve a reward that balances:

- longer lifetimes
- a target bias `T_Z / T_X`

The RL refinement notebook is a follow-on layer for the same controls:

- it starts from a physics-informed seed taken from the tutorial notebook defaults
- it uses PPO to search directly over the physical `g_2` and `epsilon_d` controls instead of depending on another optimizer
- it uses a compact Gaussian policy, PPO-style clipped updates, entropy regularization, and a replay buffer
- it keeps the cat-qubit control variables in the same physical form used in the challenge notebooks: `g_2` and `epsilon_d`
- it evaluates candidates with the same storage-buffer Hamiltonian and loss channels used in the challenge resources notebook
- it extracts `T_X` and `T_Z` from logical decay curves and uses those directly inside the PPO reward

The standalone PPO GPU script packages the same workflow into a regular Python entry point:

- it avoids notebook-kernel setup entirely
- it starts from the same physics-informed seed used in the tutorial examples
- it runs PPO directly over `g_2` and `epsilon_d`
- it prints the detected JAX devices so you can confirm that CUDA is active
- it writes characterization plots and best-decay figures to an output folder

The new batched PPO runner is the cleaner parallel-search entry point:

- it keeps the same physical controls from the challenge notebooks: `g_2` and `epsilon_d`
- it keeps the same storage-buffer Hamiltonian and loss channels instead of switching to a disconnected toy model
- it exposes a modular simulator backend, so the PPO loop can use either a fast surrogate or the full Lindblad evolution path
- it computes one reward per sampled environment from the lifetime objective `0.5 * log(T_X T_Z) - lambda_eta * (log(T_Z / T_X) - log(eta_target))^2`, with the default target bias tuned toward `eta_target = 320`
- it now treats the bias window as a hard validity region by default, with `100 <= eta <= 750`, so out-of-range candidates receive a strongly invalid reward instead of merely a soft preference
- it writes plots that show both which `T_X` and `T_Z` values were explored and which best-so-far values ultimately selected the reported `g_2` and `epsilon_d`

To stay aligned with the tutorial notebooks, the RL notebook also reports simple physics diagnostics such as:

- effective two-photon rate `kappa_2 = 4 |g_2|^2 / kappa_b`
- estimated cat size `|alpha|`
- fast-buffer health checks based on the relationship between `kappa_b`, `g_2`, and `epsilon_d`

The Python script focuses on a different observable:

- it reuses the cat-stabilization Hamiltonian from the challenge notebook
- it measures decay using the expectation value of the storage-mode parity operator
- it initializes in an odd-cat parity eigenstate by default
- it extracts the decay time from the `1/e` crossing of the normalized parity signal relative to its late-time plateau
- it tunes `g_2` and `epsilon_d` to maximize the fitted parity-decay time

## Outputs

The notebook generates:

- reward versus epoch
- optimizer trajectories for all four controls
- estimated `T_X`, `T_Z`, and bias across epochs
- scatter plots of sampled controls colored by reward
- final logical decay curves for the best candidate

The RL notebook generates characterization plots showing:

- reward progression and best-so-far reward
- the Gaussian policy mean and standard deviation over time
- sampled `g_2` and `epsilon_d` controls, colored by reward
- metric tradeoffs such as `T_X` versus `T_Z` and fidelity versus leakage
- histograms of derived cat-qubit diagnostics such as estimated cat size and the fast-buffer ratio
- explicit logical `T_X` and `T_Z` decay plots for the best PPO-refined candidate

The standalone PPO script writes the same style of outputs:

- PPO reward progression and policy statistics
- sampled-control tradeoff plots
- cat-size and fast-buffer histograms
- best-candidate `T_X` and `T_Z` decay curves

The batched PPO runner adds a more explicit training record:

- best-so-far trajectories for `Re(g_2)`, `Im(g_2)`, `Re(epsilon_d)`, and `Im(epsilon_d)`
- a combined `T_X` / `T_Z` plot that overlays all tested lifetime values with the best-so-far lifetime traces
- snapshot decay plots every configured interval, defaulting to every 100 epochs
- the same output structure whether you use the surrogate backend or the full Lindblad backend

These graphs are meant to help a non-specialist answer a simple question: not just "did the optimizer improve things?" but also "how did it explore and why does the chosen point look physically reasonable?"

The parity script prints the best fitted parity-decay time and writes plots showing:

- best parity-decay lifetime per epoch
- optimizer mean trajectories for `g_2` and `epsilon_d`
- sampled controls colored by fitted parity-decay time
- the best parity trace with its physically constrained decay model and `1/e` extraction

## How to run

1. Install the repository requirements from the root:
   `pip install -r requirements.txt`
2. Open `team-piqasso/four-parameter-optimizer.ipynb`.
3. Run the cells from top to bottom.

To explore the RL refinement workflow:

1. Install the same repository requirements from the root:
   `pip install -r requirements.txt`
2. Open `team-piqasso/rl_refinement_notebook.ipynb`.
3. Run the cells from top to bottom.
4. Adjust the physics seed, PPO settings, or reward weights if you want to steer the search differently.

The notebook includes a GPU setup cell and is written to take advantage of JAX-backed `dynamiqs` execution when a GPU is available. The PPO math is vectorized, and the simulation code stays aligned with the challenge notebooks so it is easy to compare with the rest of the project.

To run the standalone PPO search on GPU from the repository root, use the WSL environment that has CUDA-enabled JAX installed:

```bash
wsl
source ~/yquantum-gpu-venv/bin/activate
cd /mnt/c/Users/alexw/Downloads/YQuantum/Piqasso-YQuantumSubmission-Alice-Bob-Challenge
python team-piqasso/ppo_cat_gpu_search.py
```

Example quick smoke test:

```bash
python team-piqasso/ppo_cat_gpu_search.py --epochs 1 --batch-size 1 --replay-sample-size 1 --ppo-epochs 1 --na 10 --nb 4 --nsave 24 --x-tfinal 1.0 --z-tfinal 8.0 --output-dir team-piqasso/outputs/ppo_gpu_smoke_test
```

If the script prints `JAX devices: [CudaDevice(id=0)]`, it is using the GPU path successfully.

To run the new parallelizable PPO runner from the repository root:

```bash
python team-piqasso/ppo_batched_parallel_search.py
```

To run the same file on GPU from your CUDA-enabled WSL environment:

```bash
wsl
source ~/yquantum-gpu-venv/bin/activate
cd /mnt/c/Users/alexw/Downloads/YQuantum/Piqasso-YQuantumSubmission-Alice-Bob-Challenge
python team-piqasso/ppo_batched_parallel_search.py --backend lindblad --epochs 1000 --batch-size 12 --replay-sample-size 64 --snapshot-every 100
```

Useful runtime logging flags:

- `--log-every 1` prints the status of every epoch
- `--log-candidates` also prints each candidate control point in the batch
- `--quiet` suppresses the added progress prints if you want a cleaner run later

Useful bias-range flags:

- `--eta-min 100`
- `--eta-max 750`
- `--target-bias 320`
- `--lambda-eta 2.0`

Useful speed flags:

- `--eval-x-tfinal 1.0`
- `--eval-z-tfinal 24.0`
- `--eval-nsave 24`

These speed flags only affect the training-time PPO evaluations. The saved decay snapshots still use the larger `--x-tfinal`, `--z-tfinal`, and `--nsave` values so the final plots remain easier to inspect.

Quick surrogate smoke test:

```bash
python team-piqasso/ppo_batched_parallel_search.py --backend surrogate --epochs 2 --batch-size 4 --replay-sample-size 4 --ppo-epochs 1 --snapshot-every 1 --output-dir team-piqasso/outputs/ppo_batched_parallel_smoke
```

Reduced Lindblad smoke test:

```bash
python team-piqasso/ppo_batched_parallel_search.py --backend lindblad --epochs 1 --batch-size 1 --replay-sample-size 1 --ppo-epochs 1 --snapshot-every 1 --na 8 --nb 3 --nsave 16 --x-tfinal 0.8 --z-tfinal 6.0 --output-dir team-piqasso/outputs/ppo_batched_parallel_lindblad_smoke
```

The new runner is intentionally aligned with the challenge notebook physics. In plain language: it does not just search over arbitrary numbers. It simulates the same storage and buffer interaction used in the challenge material, measures short-horizon logical decay signals, and scores each candidate using physically motivated rewards and penalties.

When the GPU path is active, the script startup log will show something like `JAX devices: [CudaDevice(id=0)]`. If it only shows `CpuDevice`, then the script is still running on CPU.

To run the parity-based optimizer from the repo root:

`python team-piqasso/parity_decay_optimizer.py`

Example quick run:

`python team-piqasso/parity_decay_optimizer.py --epochs 4 --population-size 4 --tfinal 5`

This tooling is meant to be easy to inspect and extend rather than a fully optimized final submission. The CMA notebook gives a broad optimizer, the parity script focuses on parity-decay lifetime, the earlier PPO script provides a direct GPU-search baseline, and the new batched PPO runner provides the cleanest modular path for parallel reward-driven search over `g_2` and `epsilon_d`.
