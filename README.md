# Q-SpiRL

This repository contains the code for the paper **"Q-SpiRL: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation,"** accepted at the **2026 IEEE International Conference on Quantum Computing and Engineering (QCE 2026)**.

Paper: https://arxiv.org/abs/2605.20801

This repository implements a grid-based robot navigation framework for comparing classical, spiking, and quantum-enhanced reinforcement learning agents under the same environment and evaluation settings.

## Overview

The project includes multiple agents trained and evaluated on the same path-planning environment:

- A DSQN-inspired spiking neural network
- A classical MLP baseline with the same architecture and a comparable number of parameters
- A quantum-enhanced DSQN with a comparable number of parameters to the non-quantum model
- A quantum-enhanced MLP baseline with a comparable number of parameters to the non-quantum model
- A tabular Q-learning baseline

## Implemented Agents

- **Spiking DSQN**  
  A spiking neural network used to estimate Q-values for robot navigation.

- **MLP Baseline**  
  A fully connected neural network matched to the DSQN architecture in depth and parameter count for fair comparison.

- **Quantum DSQN**  
  A quantum-enhanced variant of the spiking DSQN that integrates parameterized quantum circuits.

- **Quantum MLP**  
  A quantum-enhanced version of the classical MLP baseline.

- **Tabular Q-Learning**  
  A standard Q-table implementation used as a classical baseline.

## Training

### Train All Neural Network Agents

To train and save all neural models, including the spiking, MLP, and quantum variants, run the following command on an HPC system:

```bash
sbatch train_all.sh
```

## Citation

Please cite the following paper when using this repository:

```bibtex
@article{qspirl2026,
  title   = {{Q-SpiRL}: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation},
  author  = {Altrabulsi, Mohamed Khair and Innan, Nouhaila and Marchisio, Alberto and Kashif, Muhammad and Shafique, Muhammad},
  journal = {arXiv preprint arXiv:2605.20801},
  year    = {2026}
}
```

The citation will be updated with the official proceedings information after publication.
