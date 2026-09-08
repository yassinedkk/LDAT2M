# Markov Decision Processes for Snakes and Ladders

A Python implementation of sequential decision-making methods for a stochastic Snakes and Ladders environment. The project compares model-based planning, empirical simulation and model-free reinforcement learning.

**[Read the complete project report](report.pdf)**

## Project overview

The game is represented as a Markov Decision Process (MDP) with 15 board states. At each non-terminal state, the player chooses one of three dice:

- **security die**: moves 0 or 1 square and does not activate traps;
- **normal die**: moves 0, 1 or 2 squares and activates traps with probability 0.5;
- **risky die**: moves 0, 1, 2 or 3 squares and always activates traps.

The objective is to minimize the expected number of turns required to reach the terminal square. The environment supports restart, penalty, prison and bonus traps, as well as an optional circular-board rule.

## Methods

- Markov Decision Processes;
- Bellman optimality equation;
- value iteration;
- Monte Carlo simulation;
- optimal-policy extraction;
- comparison with fixed and random policies;
- tabular Q-learning with epsilon-greedy exploration;
- empirical validation of theoretical state values.

## Main workflow

1. Construct the action-dependent transition probabilities.
2. Apply value iteration until the largest value update is below `1e-6`.
3. Extract the minimum-cost die for every state.
4. Simulate repeated games under the optimal policy.
5. Compare simulated and theoretical costs.
6. Compare the optimal policy with fixed-die and random strategies.
7. Train a Q-learning agent and compare its learned costs with value iteration.

## Repository structure

```text
portfolio/markov-decision-processes/
├── README.md
├── requirements.txt
├── report.pdf
└── snake_game.py
```

- `snake_game.py` contains the original implementation used for the coursework.
- `report.pdf` contains the original group report with all student identification numbers removed.
- `requirements.txt` lists the Python dependencies.

## Run the project

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python snake_game.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The script prints the converged policies and opens figures comparing theoretical costs, simulated costs, alternative strategies and Q-learning.

## Skills demonstrated

Python, NumPy, Matplotlib, stochastic modelling, dynamic programming, reinforcement learning, simulation, numerical convergence and statistical visualization.

## Authors

- Gaetan Berlaimont
- Benoit Henrion
- Yassine Zeamari

Academic group project for LINFO2275 - Data Mining and Decision Making, UCLouvain, 2024-2025.
