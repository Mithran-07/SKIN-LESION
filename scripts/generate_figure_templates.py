"""
Publication Figures Generator.

Creates reusable plotting templates for paper figures.
Saves as SVG, PDF, and PNG.
Does NOT generate dummy experimental results, only functions/templates.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def setup_plot_style():
    """Configure matplotlib for publication quality."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def save_fig(fig, path: str):
    """Save figure in multiple formats."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{p}.pdf", bbox_inches='tight')
    fig.savefig(f"{p}.svg", bbox_inches='tight')
    fig.savefig(f"{p}.png", bbox_inches='tight')

def plot_roc_curve_template(output_path: str = "figures/templates/roc_curve"):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)
    plt.close(fig)

def plot_pr_curve_template(output_path: str = "figures/templates/pr_curve"):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)
    plt.close(fig)

def plot_confusion_matrix_template(output_path: str = "figures/templates/confusion_matrix"):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    # Empty heatmap template
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    save_fig(fig, output_path)
    plt.close(fig)

def plot_training_curves_template(output_path: str = "figures/templates/training_curves"):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss / Accuracy')
    ax.set_title('Training Progress')
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)
    plt.close(fig)

def generate_all_templates():
    print("Generating publication figure templates...")
    plot_roc_curve_template()
    plot_pr_curve_template()
    plot_confusion_matrix_template()
    plot_training_curves_template()
    print("Templates saved to figures/templates/")

if __name__ == "__main__":
    generate_all_templates()
