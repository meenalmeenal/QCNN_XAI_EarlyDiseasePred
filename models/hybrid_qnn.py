import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # angle encoding — first 4 features
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    # variational layers
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer][i], wires=i)
            qml.RZ(weights[layer][i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class HybridQNN(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        # classical pre-processing layer
        self.pre = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, N_QUBITS),
            nn.Tanh()  # squash to [-1,1] for angle encoding
        )
        # quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS) * 0.1
        )
        # classical post-processing
        self.post = nn.Sequential(
            nn.Linear(N_QUBITS, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.pre(x)
        q_out = torch.stack([
            torch.stack(quantum_circuit(x[i], self.q_weights))
            for i in range(x.shape[0])
        ]).float()  # add .float() here
        return self.post(q_out).squeeze()


def train_qnn(model, X_train, y_train, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model