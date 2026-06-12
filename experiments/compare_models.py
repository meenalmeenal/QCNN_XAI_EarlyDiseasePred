import torch
import numpy as np
import json
import os
from utils.preprocess import load_heart_data, preprocess
from utils.metrics import evaluate, print_metrics
from models.classical_mlp import ClassicalMLP, train_mlp
from models.hybrid_qnn import HybridQNN, train_qnn

def run():
    # load data
    df = load_heart_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # --- Classical MLP ---
    print("\nTraining Classical MLP...")
    mlp = ClassicalMLP(input_dim=13)
    mlp = train_mlp(mlp, X_train, y_train, epochs=100)
    
    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        mlp_probs = mlp(X_t).numpy()
        mlp_preds = (mlp_probs > 0.5).astype(int)
    
    mlp_metrics, mlp_cm = evaluate(y_test, mlp_preds, mlp_probs)
    print_metrics("Classical MLP", mlp_metrics)

    # --- Hybrid QNN ---
    print("\nTraining Hybrid QNN...")
    qnn = HybridQNN(input_dim=13)
    qnn = train_qnn(qnn, X_train, y_train, epochs=100)
    
    qnn.eval()
    with torch.no_grad():
        qnn_probs = qnn(X_t).numpy()
        qnn_preds = (qnn_probs > 0.5).astype(int)
    
    qnn_metrics, qnn_cm = evaluate(y_test, qnn_preds, qnn_probs)
    print_metrics("Hybrid QNN", qnn_metrics)

    # save results
    os.makedirs('results', exist_ok=True)
    results = {"classical_mlp": mlp_metrics, "hybrid_qnn": qnn_metrics}
    # convert to serializable
    for m in results:
        for k in results[m]:
            if results[m][k] is not None:
                results[m][k] = round(float(results[m][k]), 4)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/metrics.json")

if __name__ == "__main__":
    run()