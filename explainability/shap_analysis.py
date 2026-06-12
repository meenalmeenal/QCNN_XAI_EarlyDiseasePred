import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def explain_mlp(model, X_train, X_test, feature_names):
    model.eval()
    
    def predict(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            out = model(t).numpy()
            return out.reshape(-1)  # ensure shape (n,)
    
    explainer = shap.KernelExplainer(predict, X_train[:50])
    shap_values = explainer.shap_values(X_test[:20])
    
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test[:20],
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('results/plots/shap_mlp.png', dpi=150)
    plt.close()
    print("SHAP plot saved to results/plots/shap_mlp.png")

def explain_qnn(model, X_train, X_test, feature_names):
    model.eval()
    
    def predict(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            out = model(t).numpy()
            return out.reshape(-1)  # ensure shape (n,)
    
    explainer = shap.KernelExplainer(predict, X_train[:50])
    shap_values = explainer.shap_values(X_test[:20])
    
    plt.figure()
    shap.summary_plot(shap_values, X_test[:20],
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('results/plots/shap_qnn.png', dpi=150)
    plt.close()
    print("SHAP plot saved to results/plots/shap_qnn.png")