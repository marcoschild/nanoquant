# nanoquant

🚀 A minimalist hybrid neural network framework with built-in quantum gate simulation.  
Inspired by micrograd and tinygrad — but quantum-powered.

## Features
- Reverse-mode autodiff (`Tensor`)
- Quantum state evolution using gates (H, X, CNOT, RY(θ), etc.)
- Hybrid neural models: QuantumLayer + ClassicalLinear + Activations
- XOR classifier demo included

## Quick Start

```bash
git clone https://github.com/yourusername/nanoquant.git
cd nanoquant
pip install -r requirements.txt
python examples/xor_quantum_example.py
```

## Example Output

```
Epoch 0, Loss: 0.2561
...
Final predictions:
Input: [0, 0], Prediction: 0.0312
Input: [0, 1], Prediction: 0.9503
...
```

## License
MIT
