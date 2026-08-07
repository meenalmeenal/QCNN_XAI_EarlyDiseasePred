import torch,json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.preprocess import load_heart_data, preprocess
from models.classical_mlp import ClassicalMLP, train_mlp
from models.hybrid_qnn import HybridQNN, train_qnn
from explainability.shap_analysis import explain_mlp, explain_qnn
from experiments.compare_models import run_comparison

import torch, numpy as np, random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

if __name__ == "__main__":
    print("Step 1: Loading and Preprocessing Dataset...")
    df = load_heart_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("\nStep 2: Training Classical MLP Baseline...")
    mlp = ClassicalMLP(input_dim=13)
    mlp, mlp_losses = train_mlp(mlp, X_train, y_train, epochs=100)

    print("\nStep 3: Training Hybrid Quantum Neural Network...")
    qnn = HybridQNN(input_dim=13)
    qnn, qnn_losses = train_qnn(qnn, X_train, y_train, epochs=100)

    print("\nStep 3b: Saving Loss Curve...")
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(mlp_losses, label='Classical MLP', linewidth=2)
    plt.plot(qnn_losses, label='Hybrid QNN', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('results/plots/loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/plots/loss_curve.png")

    print("\nStep 4: Generating SHAP Explanations...")
    explain_mlp(mlp, X_train, X_test, FEATURE_NAMES)
    explain_qnn(qnn, X_train, X_test, FEATURE_NAMES)

    print("\nStep 5: Metric Logging and Final Comparison...")
    run_comparison(mlp, qnn, X_test, y_test)

    print("\nStep 6: Generating Metrics Bar Chart...")
    with open('results/metrics.json') as f:
        results = json.load(f)
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    mlp_scores = [results['classical_mlp'][k] * 100 for k in keys]
    qnn_scores = [results['hybrid_qnn'][k] * 100 for k in keys]

    x = np.arange(len(metric_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mlp_scores, width, label='Classical MLP', color='#2a78d6', zorder=3)
    bars2 = ax.bar(x + width/2, qnn_scores, width, label='Hybrid QNN', color='#eb6834', zorder=3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, color='#2a78d6')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, color='#eb6834')
    ax.set_xlabel('Metrics', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Classical MLP vs Hybrid QNN — Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(70, 100)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved metrics_comparison.png")

    print("\nEverything completed successfully. Check results/metrics.json and results/plots/")

    