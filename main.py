import torch
import numpy as np
from utils.preprocess import load_heart_data, preprocess
from models.classical_mlp import ClassicalMLP, train_mlp
from models.hybrid_qnn import HybridQNN, train_qnn
from experiments.compare_models import run
from explainability.shap_analysis import explain_mlp, explain_qnn

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

if __name__ == "__main__":
    # run comparison
    run()

    # explainability
    print("\nGenerating SHAP explanations...")
    df = load_heart_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    X_t = torch.tensor(X_test, dtype=torch.float32)

    mlp = ClassicalMLP(input_dim=13)
    mlp = train_mlp(mlp, X_train, y_train, epochs=100)
    explain_mlp(mlp, X_train, X_test, FEATURE_NAMES)

    qnn = HybridQNN(input_dim=13)
    qnn = train_qnn(qnn, X_train, y_train, epochs=100)
    explain_qnn(qnn, X_train, X_test, FEATURE_NAMES)

    print("\nAll done! Check results/ folder.")