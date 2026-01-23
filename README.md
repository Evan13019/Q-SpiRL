# QRL-Robot-Navigation

This repository implements a grid-based path navigation task inspired by Kumar et al (2025) [DSQN paper]. The goal is to compare classical, spiking, and quantum-enhanced reinforcement learning agents on the same navigation problem.

## Overview

The project includes multiple agents trained and evaluated on the same path-planning environment:

- A DSQN-inspired spiking neural network
- A spiking DSQN-style baseline
- A classical MLP baseline with the same architecture and number of parameters as DSQN
- A quantum-enhanced DSQN (comprable number of parameters as non-quantum)
- A quantum-enhanced MLP baseline (comprable number of parameters as non-quantum)
- A tabular Q-learning baseline

## Implemented Agents
- **Spiking DSQN**  
  A spiking neural network inspired by the DSQN paper, used to estimate Q-values for navigation.

- **MLP Baseline**  
  A fully connected neural network matched to the DSQN architecture in depth and parameter count for fair comparison.

- **Quantum DSQN**  
  A quantum-enhanced variant of the spiking DSQN, integrating parameterized quantum circuits.

- **Quantum MLP**  
  A quantum-enhanced version of the classical MLP baseline.

- **Tabular Q-Learning**  
  A standard Q-table implementation used as a classical baseline.

## Training

### Train all neural network agents
To train and save all neural models (spiking, MLP, and quantum variants), run (on HPC):

```bash
sbatch train_all.sh
