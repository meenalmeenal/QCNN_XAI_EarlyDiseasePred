import torch
import numpy as np
from utils.preprocess import load_heart_data, preprocess
from models.classical_mlp import ClassicalMLP, train_mlp
from models.hybrid_qnn import HybridQNN, train_qnn
from explainability.shap_analysis import explain_mlp, explain_qnn
from experiments.compare_models import run_comparison

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

if __name__ == "__main__":
    print("Step 1: Loading and Preprocessing Dataset...")
    df = load_heart_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("\nStep 2: Training Classical MLP Baseline...")
    mlp = ClassicalMLP(input_dim=13)
    mlp = train_mlp(mlp, X_train, y_train, epochs=100)

    print("\nStep 3: Training Hybrid Quantum Neural Network...")
    qnn = HybridQNN(input_dim=13)
    qnn = train_qnn(qnn, X_train, y_train, epochs=100)

    print("\nStep 4: Generating SHAP Explanations...")
    explain_mlp(mlp, X_train, X_test, FEATURE_NAMES)
    explain_qnn(qnn, X_train, X_test, FEATURE_NAMES)

    print("\nStep 5: Metric Logging and Final Comparison...")
    run_comparison(mlp, qnn, X_test, y_test)

    print("\nEverything completed successfully. Check results/metrics.json and results/plots/")