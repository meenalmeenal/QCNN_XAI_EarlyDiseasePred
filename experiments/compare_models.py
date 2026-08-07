import torch
import numpy as np
import json
import os
from utils.metrics import evaluate, print_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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
    # --- Confusion matrices ---
    os.makedirs('results/plots', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in [(axes[0], mlp_preds, 'Classical MLP'),
                              (axes[1], qnn_preds, 'Hybrid QNN')]:
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease']).plot(ax=ax, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- ROC curve ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for probs, name in [(mlp_probs, 'Classical MLP'), (qnn_probs, 'Hybrid QNN')]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Classical MLP vs Hybrid QNN')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig('results/plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Saved confusion_matrix.png and roc_curve.png to results/plots/")