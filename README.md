# Sum of Three Cubes

## Overview
This repository implements an environment for the Sum of Cubes problem and tests it under various conditions. The PPO (Proximal Policy Optimization) implementation is based on the Arena 2.3 PPO lecture's Jupyter notebook (https://arena3-chapter2-rl.streamlit.app/[2.3]_PPO).

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the experiment using PPOArgs
2. Select your target environment
3. Start the training process

## Configuration Parameters

The PPOArgs class supports the following parameters:

- `total_timesteps`: Total number of actions the experiment will take
- `num_envs`: Number of parallel environments to run
- `max_k`: Maximum value for k (where k = x³ + y³ + z³)
- `learning_rate`: Learning rate for the optimization

### Advantage Calculation Parameters
- `gamma`
- `gae_lambda`

### Loss Calculation Parameters
- `clip_coef`
- `ent_coef`
- `vf_coef`

Note: Different environments may require different parameter settings. You can adjust these in the PPOArgs class within PPO.py.
You can also develop your own environment.

