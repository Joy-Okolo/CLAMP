This repository contains a research implementation of federated learning (FL) that evaluates multiple strategies under realistic heterogeneous client conditions on three benchmark datasets:
- MNIST  
- Fashion-MNIST  
- CIFAR-10  

The code is written in PyTorch and is designed for research focusing on:
- True convergence (no fixed round limit, early stopping based on accuracy plateau)
- Straggler tracking and participation analysis
- Computational and communication efficiency
- Dynamic layer adaptation for heterogeneous devices

 ## 1. Main Ideas
The project compares several federated learning strategies:
- FedAvg – standard federated averaging with full model training  
- FedPMT – partial model training with a fixed number of layers updated  
- FedDrop – randomly drops layers to simulate lighter local computation  
- DLA (Dynamic Layer Adaptation) – adapts how many layers each client trains based on its speed and performance

Key components in the code include:
- GlobalConvergenceTracker – detects when global training has truly converged  
- ResearchGradeStragglerTracker – monitors slow clients (stragglers), inclusion rates, and accommodation efficiency  
- ResearchGradeEfficiencyTracker – estimates FLOPs, memory usage, energy, and communication cost per round and per strategy  
- Dataset creation functions for federated versions of MNIST, Fashion-MNIST, and CIFAR-10  
- Training loops that can run “until convergence” with a safety limit on the maximum number of rounds

## 2. Datasets
The framework is designed to support three datasets:
- MNIST – 28×28 grayscale digits  
- Fashion-MNIST – 28×28 grayscale fashion items  
- CIFAR-10 – 32×32 RGB natural images  

Each dataset can be split across clients in:
- IID mode (uniform random split)  
- Non-IID mode (clients biased toward certain classes)

## 3. Dependencies
- Python 3.8 or higher  
- PyTorch and torchvision  
- NumPy  
- Matplotlib  

Example installation (adjust versions as needed):
bash
pip install torch torchvision
pip install numpy matplotlib

## 5. How to Run
Below is a general pattern. Adjust script names to match your files.

CIFAR-10 “no round limit” experiment
python clamp_cifar10.py

This will create federated CIFAR-10 clients
Run FedAvg, FedPMT, FedDrop, and DLA
Train until a target test accuracy (e.g., 75%) is reached or a safety round limit is hit
Track FLOPs, communication, stragglers, and convergence information
Save plots to ./plots and metrics to ./results

For MNIST and Fashion-MNIST
python clamp_mnist.py
python clamp_fmnist.py

Typical configurable parameters (either inside the script or via arguments):
num_clients – number of simulated clients (e.g., 50)
max_local_epochs – local training epochs per round
iid – whether to use IID or non-IID data split
strategy – which FL algorithm to run (fedavg, fedpmt, feddrop, dla)

6. Outputs
The experiments produce:
JSON results (e.g. in ./results/), including:
Test accuracy and loss per round
Rounds to convergence
FLOPs, communication cost, and energy per round
Straggler statistics and inclusion rates

Plots (e.g. in ./plots/), such as:
Accuracy vs round
Rounds to convergence per strategy
Communication and computational efficiency comparisons
Plots are saved in PNG format with timestamps in their filenames.
