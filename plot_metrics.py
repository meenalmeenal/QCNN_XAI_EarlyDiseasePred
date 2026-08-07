import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
import json
with open('results/metrics.json') as f:
    results = json.load(f)

# Check your utils/metrics.py to confirm these exact key names
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
mlp_scores = [results['classical_mlp'][k]*100 for k in ['accuracy','precision','recall','f1','auc','specificity']]
qnn_scores = [results['hybrid_qnn'][k]*100 for k in ['accuracy','precision','recall','f1','auc','specificity']]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, mlp_scores, width, label='Classical MLP', color='#2a78d6', zorder=3)
bars2 = ax.bar(x + width/2, qnn_scores, width, label='Hybrid QNN',    color='#eb6834', zorder=3)

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
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(70, 100)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

os.makedirs('results/plots', exist_ok=True)
plt.tight_layout()
plt.savefig('results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to results/plots/metrics_comparison.png")

