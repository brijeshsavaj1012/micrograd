# Autograd-like Implementation in Python

This repository demonstrates:
1. **A minimal `Value` class** that tracks data, gradients, and operations for automatic differentiation.  
2. **Graph visualization** using Graphviz to show the computation graph.  
3. **A simple MLP** (Multi-Layer Perceptron) to showcase training with forward and backward passes.

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
