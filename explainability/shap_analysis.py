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
            return model(t).cpu().numpy()
            
    explainer = shap.KernelExplainer(predict, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_test[:10])
    
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values
        
    if len(shap_values_to_plot.shape) == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 0]
    elif len(shap_values_to_plot.shape) == 2 and shap_values_to_plot.shape[1] != len(feature_names):
        shap_values_to_plot = shap_values_to_plot.squeeze()

    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test[:10],
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
            return model(t).cpu().numpy()
            
    explainer = shap.KernelExplainer(predict, shap.sample(X_train, 30))
    shap_values = explainer.shap_values(X_test[:10])
    
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values
        
    if len(shap_values_to_plot.shape) == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 0]
    elif len(shap_values_to_plot.shape) == 2 and shap_values_to_plot.shape[1] != len(feature_names):
        shap_values_to_plot = shap_values_to_plot.squeeze()
        
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test[:10],
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('results/plots/shap_qnn.png', dpi=150)
    plt.close()
    print("SHAP plot saved to results/plots/shap_qnn.png")