# squeezing_by_control

Code to find optimized time-dependent control pulses in Rydberg-atom arrays to generate spin‑squeezed states by unitary evolution. Implementation uses JAX for numeric kernels, Dynamiqs for time evolution, and Optax for optimization helpers.

Repository
- GitHub: https://github.com/edison651/squeezing_by_control
- Owner: edison651
- Default branch: main
- Current tree: https://github.com/edison651/squeezing_by_control/tree/main
- Ref: Edison S. Carrera, Harold Erbin & Grégoire Misguich (2025) Preparing spin-squeezed states in Rydberg atom arrays via quantum optimal control, https://arxiv.org/abs/2507.07875

Contents
- module_control.py — core functions and building blocks (Hamiltonians, collective operators, cost functions, time evolution helpers, analysis/plotting).
- plaympi.py — example driver that runs optimizations from multiple initial conditions in parallel using MPI (mpi4py). Edit the script to fit your environment (see notes below).

Quick summary of important functions in module_control.py
- pauli_matrix(a), sigma_j(j,a,n), col_sigmas(n, pauli), sigmaA_jxsigmaB_k(...) — construct single- and multi-spin Pauli operators.
- generate_basis, translation_operator, get_momentum_eigenstates — basis and translation/momentum operators for periodic chains.
- ring(N), real_vander_ising_H(N, vec_pos, ...) , vander_ising_H(...) — position helpers and interaction Hamiltonian builders.
- css(theta, phi, n) — collective spin coherent state.
- time_evo(b, T, H_int, control, vec) — JAX-jitted discrete-time propagator.
- wineland_cost(...) and wineland_and_length(...) — cost/squeezing metrics (Wineland parameter) used for optimization.
- penalties(...) and funs_with_penalty(...) — regularizers for pulse smoothness/amplitude/duration.
- jax_control_squeezing_mod(...) — high-level optimization routine (returns best squeezing, best params, steps used, last improvement, spin length).
- control_tevol(...), dissipation_test(...) — wrappers for Dynamiqs sesolve/mesolve simulations using piecewise-constant (pwc) controls.
- find_sq(...), find_var(...), find_angle(...) — routines to evaluate squeezing directions and angles.
- plotting helpers: plot_q_husumi, plot_q_husumi_3d
- utilities: sample_from_distribution, jacky (jackknife), entropy_half, extract_data

Dependencies (inferred)
- Python 3.8+ (3.9/3.10 recommended)
- jax, jaxlib
- numpy
- dynamiqs (imported as dynamiqs / dq)
- optax
- matplotlib
- mpi4py (for plaympi.py)

Example requirements.txt (adjust versions as needed)
- jax[cpu]
- numpy
- dynamiqs
- optax
- matplotlib
- mpi4py

Installation
1. Clone repo:
   git clone https://github.com/edison651/squeezing_by_control.git
   cd squeezing_by_control

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate

3. Install dependencies (example):
   pip install --upgrade pip
   pip install "jax[cpu]" numpy dynamiqs optax matplotlib mpi4py

Running the example (plaympi.py)
- plaympi.py is an example driver that:
  - sets system size and optimization hyperparameters,
  - constructs h0 and projected spin operators via momentum eigenstates,
  - calls qc.jax_control_squeezing_mod(...) to optimize from an initial parameter sample,
  - gathers results across MPI ranks and writes results to a text file.

Minimal run (single process)
- You can call the optimization function directly from Python. See function docstrings in module_control.py for parameters.

MPI run (multiple initial conditions)
- To run with 4 processes:
  mpirun -np 4 python plaympi.py
  or
  mpiexec -n 4 python plaympi.py

Important note about output path
- plaympi.py currently writes results to a hard-coded path:
  '/home/ipht/ecarrera/qcontrol/data/itneverends/...'
  Edit the filepath variable in plaympi.py (near where filepath2 is assigned) to a directory on your machine, for example:
  filepath2 = f'results/params_n{n}_T{total_t}_p{para_per_gate}.txt'

Output format
- plaympi.py saves:
  - A header with learned hyperparameters
  - For each MPI rank: "Best squeezing, steps, dif, Spin length" and numeric values
  - A saved table with columns: time, Z-control, X-control, H0 (via numpy.savetxt)
- Use module_control.extract_data(filepath) to parse the saved file.

Reproducibility tips
- Save the full parameter dictionary (learn_params) and initial params for each run.
- Save the git commit hash used for the run:
  git rev-parse --short HEAD > results/<run>/git-hash.txt

Troubleshooting & tips
- If JAX or Dynamiqs errors occur, check version compatibility between jax, jaxlib, and dynamiqs.
- If MPI + GPU: ensure each MPI process uses a distinct GPU (e.g., set CUDA_VISIBLE_DEVICES per process) or run one MPI rank per GPU.
- If the optimization stalls, try different ini_learning_rate or reduce learning_steps.

