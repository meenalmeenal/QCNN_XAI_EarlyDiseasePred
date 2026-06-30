import torch
import numpy as np
import json
import os
from utils.metrics import evaluate, print_metrics

def run_comparison(mlp_model, qnn_model, X_test, y_test):
    mlp_model.eval()
    qnn_model.eval()
    
    X_t = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        mlp_probs = mlp_model(X_t).cpu().numpy().flatten()
        mlp_preds = (mlp_probs >= 0.5).astype(int)
        
        qnn_probs = qnn_model(X_t).cpu().numpy().flatten()
        qnn_preds = (qnn_probs >= 0.5).astype(int)
    
    mlp_metrics, mlp_cm = evaluate(y_test, mlp_preds, mlp_probs)
    print_metrics("Classical MLP", mlp_metrics)
    
    qnn_metrics, qnn_cm = evaluate(y_test, qnn_preds, qnn_probs)
    print_metrics("Hybrid QNN", qnn_metrics)
    
    os.makedirs('results', exist_ok=True)
    results = {"classical_mlp": mlp_metrics, "hybrid_qnn": qnn_metrics}
    
    for m in results:
        for k in results[m]:
            if results[m][k] is not None:
                results[m][k] = round(float(results[m][k]), 4)
                
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nResults saved to results/metrics.json")