"""
Visualization Module for Fraud Detection Results
Generates all required visualizations for the project
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class FraudVisualizer:
    """
    Handles all visualizations for the fraud detection project
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize visualizer
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_history(self, history, save=True):
        """
        Plot training and validation loss over epochs
        
        Args:
            history: Keras training history object
            save (bool): Whether to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, 'training_history.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
    
    def plot_reconstruction_error_distribution(self, errors, labels, threshold=None, save=True):
        """
        THE MONEY SHOT: Histogram showing separation between normal and fraud
        
        Args:
            errors (np.array): Reconstruction errors
            labels (np.array): True labels (0=normal, 1=fraud)
            threshold (float): Decision threshold (optional)
            save (bool): Whether to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Separate errors by class
        normal_errors = errors[labels == 0]
        fraud_errors = errors[labels == 1]
        
        # Plot histograms
        ax.hist(normal_errors, bins=50, alpha=0.7, label='Normal Transactions', 
                color='#2E86AB', edgecolor='black')
        ax.hist(fraud_errors, bins=50, alpha=0.7, label='Fraudulent Transactions', 
                color='#A23B72', edgecolor='black')
        
        # Add threshold line
        if threshold is not None:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold = {threshold:.4f}')
        
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('The "Truth Filter": Reconstruction Error Distribution', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Normal: μ={normal_errors.mean():.4f}, σ={normal_errors.std():.4f}\n'
        stats_text += f'Fraud: μ={fraud_errors.mean():.4f}, σ={fraud_errors.std():.4f}'
        ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save:
            filepath = os.path.join(self.output_dir, 'reconstruction_error_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save=True):
        """
        Plot confusion matrix
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            save (bool): Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add percentages
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        metrics_text = f'TN: {tn} ({tn/total*100:.1f}%)\n'
        metrics_text += f'FP: {fp} ({fp/total*100:.1f}%)\n'
        metrics_text += f'FN: {fn} ({fn/total*100:.1f}%)\n'
        metrics_text += f'TP: {tp} ({tp/total*100:.1f}%)'
        
        plt.text(1.5, 0.5, metrics_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, errors, save=True):
        """
        Plot Precision-Recall curve (more relevant for imbalanced data)
        
        Args:
            y_true (np.array): True labels
            errors (np.array): Reconstruction errors (used as scores)
            save (bool): Whether to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, errors)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        ax.fill_between(recall, precision, alpha=0.3)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save:
            filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
        
        return pr_auc
    
    def plot_feature_explainability(self, input_sample, reconstructed_sample, 
                                    feature_names=None, top_n=10, save=True):
        """
        THE XAI COMPONENT: Show which features triggered the fraud alert
        
        Args:
            input_sample (np.array): Original input features
            reconstructed_sample (np.array): Reconstructed features
            feature_names (list): Names of features (optional)
            top_n (int): Show top N features with highest error
            save (bool): Whether to save the plot
        """
        # Calculate absolute error per feature
        feature_errors = np.abs(input_sample - reconstructed_sample)
        
        # Sort by error magnitude
        sorted_indices = np.argsort(feature_errors)[::-1][:top_n]
        
        # Prepare data for plotting
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(input_sample))]
        
        selected_features = [feature_names[i] for i in sorted_indices]
        selected_errors = feature_errors[sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        bars = ax.barh(selected_features, selected_errors, color='#C73E1D', edgecolor='black')
        
        ax.set_xlabel('Reconstruction Error (Absolute Difference)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature-Wise Explainability: Why Was This Flagged?', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, selected_errors)):
            ax.text(error, i, f' {error:.4f}', va='center', fontsize=9)
        
        plt.gca().invert_yaxis()  # Highest error at top
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_explainability.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        plt.close()
    
    def print_classification_report(self, y_true, y_pred, threshold):
        """
        Print detailed classification metrics
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            threshold (float): Decision threshold used
        """
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(f"\nThreshold used: {threshold:.6f}\n")
        
        print(classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Fraud'],
                                   digits=4))
        
        # Additional metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nKey Metrics:")
        print(f"  Precision: {precision:.4f} (Of flagged transactions, {precision*100:.2f}% were actually fraud)")
        print(f"  Recall:    {recall:.4f} (Detected {recall*100:.2f}% of all fraud cases)")
        print(f"  F1-Score:  {f1:.4f} (Harmonic mean of precision and recall)")
        
        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Negatives (TN):  {tn:6d} - Correctly identified normal transactions")
        print(f"  False Positives (FP): {fp:6d} - Normal transactions incorrectly flagged as fraud")
        print(f"  False Negatives (FN): {fn:6d} - Fraud transactions missed")
        print(f"  True Positives (TP):  {tp:6d} - Correctly detected fraud transactions")

if __name__ == "__main__":
    print("\nTesting FraudVisualizer...")
    
    # Test with dummy data
    visualizer = FraudVisualizer(output_dir='outputs')
    
    # Generate dummy reconstruction errors
    np.random.seed(42)
    normal_errors = np.random.exponential(0.02, 1000)
    fraud_errors = np.random.exponential(0.15, 50)
    errors = np.concatenate([normal_errors, fraud_errors])
    labels = np.concatenate([np.zeros(1000), np.ones(50)])
    
    threshold = np.percentile(normal_errors, 95)
    
    print(f"\nGenerating sample visualizations...")
    visualizer.plot_reconstruction_error_distribution(errors, labels, threshold, save=False)
    
    print("\n✨ Visualizer tested successfully!")
