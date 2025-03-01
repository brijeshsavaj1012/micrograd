# Micrograd (Autograd-like Implementation in Python)
This repository contains a simple automatic differentiation engine implemented from scratch using a Value class. It also includes a minimal Multilayer Perceptron (MLP) for demonstration, showing how to perform forward and backward passes as well as basic optimization steps

## Installation
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt  # or: pip install torch graphviz matplotlib
```
> Make sure [Graphviz](https://graphviz.org/download/) is installed on your system.

## Usage
1. **Run the code**:  
   ```bash
   python main.py
   ```
2. **Train the MLP**:  
   - A simple training loop demonstrates gradient updates and parameter optimization.

## Key Files
- **`Value` class**: Implements custom autograd (supports `+`, `-`, `*`, `/`, `exp`, `tanh`, etc.).
- **`draw_dot`** and **`trace`**: Generate computation graph visualizations.
- **MLP** (with `Neuron` and `Layer`): A minimal neural network built on the `Value` class.
