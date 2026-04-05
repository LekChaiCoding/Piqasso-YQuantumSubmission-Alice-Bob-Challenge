# Alice & Bob x YQuantum 2026 Challenge

Challenge materials are in `challenge/`.

<<<<<<< HEAD
## Team notebook

A standalone exploratory notebook is available at `team-piqasso/four-parameter-optimizer.ipynb`.

It demonstrates a rudimentary `dynamiqs`-based optimizer for the four real control knobs underlying the complex cat-stabilization parameters:

- `Re(g_2)`
- `Im(g_2)`
- `Re(epsilon_d)`
- `Im(epsilon_d)`

The notebook estimates proxy logical lifetimes, optimizes toward a target bias, and plots reward, parameter trajectories, sampled control points, and final decay curves.

## Environment setup

This repository now includes a local Python `3.11` environment setup that works with the notebook in `challenge/1-challenge.ipynb`.

Recommended steps on Windows PowerShell:

```powershell
uv python install 3.11
uv venv --seed --python 3.11 .venv311
.\.venv311\Scripts\python -m pip install -r requirements.txt
```

After installation:

1. Open `challenge/1-challenge.ipynb`.
2. Select the interpreter or kernel from `.venv311`.
3. Run the import cells at the top of the notebook.

## Notes

- The notebook metadata shows it was originally run in Python `3.11.15`.
- This repository is now configured to use a local Python `3.11` environment at `.venv311`.
- The direct notebook dependencies are listed in `requirements.txt`.
=======
Branch for Batu.
>>>>>>> batu
