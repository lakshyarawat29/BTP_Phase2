"""
Threshold Analysis Tool
Helps find optimal threshold by analyzing precision-recall trade-offs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from data_loader import load_data
from preprocessor import DataPreprocessor
from autoencoder import FraudAutoencoder
import matplotlib
matplotlib.use('Agg')

def analyze_thresholds(errors, y_true, percentiles=range(90, 100, 1)):
    """
    Analyze different threshold values
    
    Args:
        errors: Reconstruction errors
        y_true: True labels
        percentiles: Range of percentiles to test
    """
    results = []
    
    for p in percentiles:
        threshold = np.percentile(errors, p)
        y_pred = (errors > threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        fp_rate = (y_pred[y_true == 0] == 1).sum() / (y_true == 0).sum()
        
        results.append({
            'percentile': p,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fp_rate,
            'flagged': y_pred.sum()
        })
    
    return results

def plot_threshold_analysis(results, save_path='outputs/threshold_analysis.png'):
    """Plot threshold analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    percentiles = [r['percentile'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]
    fp_rate = [r['fp_rate'] for r in results]
    
    # Plot 1: Precision, Recall, F1
    axes[0, 0].plot(percentiles, precision, 'b-', label='Precision', linewidth=2)
    axes[0, 0].plot(percentiles, recall, 'r-', label='Recall', linewidth=2)
    axes[0, 0].plot(percentiles, f1, 'g-', label='F1-Score', linewidth=2)
    axes[0, 0].set_xlabel('Threshold Percentile')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Metrics vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: False Positive Rate
    axes[0, 1].plot(percentiles, fp_rate, 'orange', linewidth=2)
    axes[0, 1].set_xlabel('Threshold Percentile')
    axes[0, 1].set_ylabel('False Positive Rate')
    axes[0, 1].set_title('False Positive Rate vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Trade-off
    axes[1, 0].plot(recall, precision, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Trade-off')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of Flagged Transactions
    flagged = [r['flagged'] for r in results]
    axes[1, 1].plot(percentiles, flagged, 'brown', linewidth=2)
    axes[1, 1].set_xlabel('Threshold Percentile')
    axes[1, 1].set_ylabel('Transactions Flagged')
    axes[1, 1].set_title('Flagged Transactions vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def print_recommendations(results):
    """Print threshold recommendations"""
    print("\n" + "="*80)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*80)
    
    # Find best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n🎯 Best F1-Score: {best_f1['f1']:.4f}")
    print(f"   Threshold: {best_f1['threshold']:.6f} ({best_f1['percentile']}th percentile)")
    print(f"   Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}")
    print(f"   FP Rate: {best_f1['fp_rate']:.2%}, Flagged: {best_f1['flagged']}")
    
    # Find high recall
    high_recall = [r for r in results if r['recall'] >= 0.9]
    if high_recall:
        best_hr = min(high_recall, key=lambda x: x['fp_rate'])
        print(f"\n🔍 High Recall Option (≥90% fraud detection):")
        print(f"   Threshold: {best_hr['threshold']:.6f} ({best_hr['percentile']}th percentile)")
        print(f"   Precision: {best_hr['precision']:.4f}, Recall: {best_hr['recall']:.4f}")
        print(f"   FP Rate: {best_hr['fp_rate']:.2%}, Flagged: {best_hr['flagged']}")
    
    # Find low FP
    low_fp = [r for r in results if r['fp_rate'] <= 0.01]
    if low_fp:
        best_lfp = max(low_fp, key=lambda x: x['recall'])
        print(f"\n✅ Low False Positive Option (≤1% FP rate):")
        print(f"   Threshold: {best_lfp['threshold']:.6f} ({best_lfp['percentile']}th percentile)")
        print(f"   Precision: {best_lfp['precision']:.4f}, Recall: {best_lfp['recall']:.4f}")
        print(f"   FP Rate: {best_lfp['fp_rate']:.2%}, Flagged: {best_lfp['flagged']}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS TOOL")
    print("="*80)
    
    # Load data and model
    print("\nLoading model and data...")
    df = load_data('data/creditcard.csv')
    
    preprocessor = DataPreprocessor(use_robust_scaler=True)
    X_train_normal, X_test, y_test = preprocessor.preprocess(df, drop_time=False)
    
    # Load trained model
    autoencoder = FraudAutoencoder(input_dim=X_test.shape[1])
    autoencoder.load_model('models/fraud_autoencoder.keras')
    
    # Calculate reconstruction errors
    print("\nCalculating reconstruction errors...")
    test_errors, _ = autoencoder.calculate_reconstruction_error(X_test)
    
    # Analyze thresholds
    print("\nAnalyzing thresholds from 90th to 99th percentile...")
    results = analyze_thresholds(test_errors, y_test, percentiles=range(90, 100, 1))
    
    # Plot results
    plot_threshold_analysis(results)
    
    # Print recommendations
    print_recommendations(results)
    
    print("\n✨ Analysis complete!")
