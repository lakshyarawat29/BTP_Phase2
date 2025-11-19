"""
Comparison: Autoencoder vs Other Anomaly Detection Methods
Shows why the autoencoder approach is superior
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_data
from preprocessor import DataPreprocessor
import time

def compare_methods(X_train, X_test, y_test):
    """
    Compare autoencoder with other anomaly detection methods
    """
    results = {}
    
    print("\n" + "="*80)
    print("COMPARING ANOMALY DETECTION METHODS")
    print("="*80)
    
    # 1. Isolation Forest
    print("\n1️⃣ Training Isolation Forest...")
    start = time.time()
    iso_forest = IsolationForest(contamination=0.002, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    y_pred_iso = (iso_forest.predict(X_test) == -1).astype(int)
    train_time_iso = time.time() - start
    
    results['Isolation Forest'] = {
        'precision': precision_score(y_test, y_pred_iso, zero_division=0),
        'recall': recall_score(y_test, y_pred_iso, zero_division=0),
        'f1': f1_score(y_test, y_pred_iso, zero_division=0),
        'train_time': train_time_iso,
        'y_pred': y_pred_iso
    }
    print(f"   ✓ Completed in {train_time_iso:.2f}s")
    
    # 2. One-Class SVM (sample for speed)
    print("\n2️⃣ Training One-Class SVM (on sample)...")
    sample_size = min(5000, len(X_train))
    X_train_sample = X_train[np.random.choice(len(X_train), sample_size, replace=False)]
    
    start = time.time()
    oc_svm = OneClassSVM(nu=0.002, kernel='rbf', gamma='scale')
    oc_svm.fit(X_train_sample)
    y_pred_svm = (oc_svm.predict(X_test) == -1).astype(int)
    train_time_svm = time.time() - start
    
    results['One-Class SVM'] = {
        'precision': precision_score(y_test, y_pred_svm, zero_division=0),
        'recall': recall_score(y_test, y_pred_svm, zero_division=0),
        'f1': f1_score(y_test, y_pred_svm, zero_division=0),
        'train_time': train_time_svm,
        'y_pred': y_pred_svm
    }
    print(f"   ✓ Completed in {train_time_svm:.2f}s")
    
    # 3. Local Outlier Factor
    print("\n3️⃣ Training Local Outlier Factor...")
    start = time.time()
    lof = LocalOutlierFactor(contamination=0.002, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    y_pred_lof = (lof.predict(X_test) == -1).astype(int)
    train_time_lof = time.time() - start
    
    results['Local Outlier Factor'] = {
        'precision': precision_score(y_test, y_pred_lof, zero_division=0),
        'recall': recall_score(y_test, y_pred_lof, zero_division=0),
        'f1': f1_score(y_test, y_pred_lof, zero_division=0),
        'train_time': train_time_lof,
        'y_pred': y_pred_lof
    }
    print(f"   ✓ Completed in {train_time_lof:.2f}s")
    
    return results

def print_comparison_table(results, autoencoder_results):
    """Print comparison table"""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time (s)':>10}")
    print("-"*80)
    
    # Add autoencoder results
    all_results = {'Deep Autoencoder (Ours)': autoencoder_results}
    all_results.update(results)
    
    for method, metrics in all_results.items():
        print(f"{method:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics.get('train_time', 0):>10.2f}")
    
    # Find best
    print("\n" + "="*80)
    best_recall = max(all_results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(all_results.items(), key=lambda x: x[1]['f1'])
    
    print(f"\n🏆 Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")
    print(f"🏆 Best F1-Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    
    print("\n💡 Key Insights:")
    print("   • Autoencoder provides explainability (feature-level errors)")
    print("   • Autoencoder can be fine-tuned with different architectures")
    print("   • Autoencoder learns complex non-linear patterns")
    print("   • Traditional methods are faster but less flexible")

if __name__ == "__main__":
    print("\nLoading data...")
    df = load_data('data/creditcard.csv')
    
    preprocessor = DataPreprocessor(use_robust_scaler=True)
    X_train_normal, X_test, y_test = preprocessor.preprocess(df, drop_time=False)
    
    # Load autoencoder results (from previous run)
    from autoencoder import FraudAutoencoder
    autoencoder = FraudAutoencoder(input_dim=X_test.shape[1])
    autoencoder.load_model('models/fraud_autoencoder.keras')
    
    test_errors, _ = autoencoder.calculate_reconstruction_error(X_test)
    threshold = np.percentile(test_errors, 95)
    y_pred_ae = (test_errors > threshold).astype(int)
    
    autoencoder_results = {
        'precision': precision_score(y_test, y_pred_ae),
        'recall': recall_score(y_test, y_pred_ae),
        'f1': f1_score(y_test, y_pred_ae),
        'train_time': 0  # Already trained
    }
    
    # Compare with other methods
    results = compare_methods(X_train_normal, X_test, y_test)
    
    # Print comparison
    print_comparison_table(results, autoencoder_results)
    
    print("\n✨ Comparison complete!")
