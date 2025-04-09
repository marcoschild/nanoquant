import numpy as np
from nanoquant.core.gates import H, CNOT
from nanoquant.core.layers import Quantumlayer, classicallinear, sigmoid
from nanoquant.core.egine import MSELoss, SGD, get_parameters, run_model
from nanoquant.core.tensor import tensor

# Define gate sequence for 2-qubit entagled state

def xor_gate_sequence():
    return[
        np.kron(H(), np.eye(2)), # Apply H to qubit 0
        CNOT()                   # CNOT between qubit 0 and qubit 1 
    ]

# Prepare training data for XOR
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [
    [0],
    [1],
    [1],
    [0]
]

# Build momdel
model = [
    Quantumlayer(num_qubits=2, gate_sequance_fn=xor_gate_sequence),
    classicallinear(in_features=2, out_features=1),
]

#Initalize loss and optimizer 

loss_fn = MSELoss()
params = get_parameters(model)
opt = SGD(params, lr=0.01)

# Training loop
for epoch in range (1000):
    total_loss = 0.0
    for xi, yi in zip(X, y):
        probs = model [0].forward(xi) # Get probabilities from quantum layer
        input_tensor = [tensor(p) for p in probs]
        output = run_model(model[1:], input_tensor) # Run through classical layers
        target = [tensor(float(t)) for t in yi]
        loss = loss_fn(output, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.data

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X):.4f}")
        

#Test 
print("\nFinal predictions:")
for xi in X:
    probs = model[0].forward(xi)
    input_tensor = [tensor(p) for p in probs]
    output = run_model(model[1:], input_tensor)
    print(f"Input: {xi}, Preidction: {output[0].data:.4f}")
