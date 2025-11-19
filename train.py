"""
Main Training Pipeline for Fraud Detection Autoencoder
Orchestrates the entire training process from data loading to evaluation
"""

import numpy as np
import os
from data_loader import load_data
from preprocessor import DataPreprocessor
from autoencoder import FraudAutoencoder
from visualizer import FraudVisualizer

def determine_threshold(errors, percentile=95):
    """
    Determine the anomaly threshold based on training errors
    
    Args:
        errors (np.array): Reconstruction errors on normal transactions
        percentile (int): Percentile to use as threshold (95 = 95th percentile)
        
    Returns:
        float: Threshold value
    """
    threshold = np.percentile(errors, percentile)
    print(f"\n🎯 Threshold set at {percentile}th percentile: {threshold:.6f}")
    return threshold

def main():
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("FRAUD DETECTION AUTOENCODER - TRAINING PIPELINE")
    print("The Truth Filter: Unsupervised Anomaly Detection")
    print("="*80)
    
    # ========== PHASE 1: DATA LOADING & PREPROCESSING ==========
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    
    # Load data
    df = load_data('data/creditcard.csv')
    
    # Preprocess with critical split
    preprocessor = DataPreprocessor(use_robust_scaler=True)
    X_train_normal, X_test, y_test = preprocessor.preprocess(
        df, 
        drop_time=False,  # Keep time column
        test_size=0.2,
        random_state=42
    )
    
    # Save the scaler
    preprocessor.save_scaler('models/scaler.pkl')
    
    input_dim = X_train_normal.shape[1]
    print(f"\n✓ Input dimension: {input_dim}")
    
    # ========== PHASE 2: BUILD & TRAIN AUTOENCODER ==========
    print("\n" + "="*80)
    print("PHASE 2: BUILDING & TRAINING AUTOENCODER")
    print("="*80)
    
    # Build model
    autoencoder = FraudAutoencoder(
        input_dim=input_dim,
        encoding_dims=[14, 7]  # 29 -> 14 -> 7 (bottleneck)
    )
    autoencoder.build_model()
    autoencoder.compile_model(
        optimizer='adam',
        learning_rate=0.001,
        loss='mse'
    )
    
    # Train model
    history = autoencoder.train(
        X_train=X_train_normal,
        X_val=X_test,  # Validation on mixed data
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    autoencoder.save_model('models/fraud_autoencoder.keras')
    
    # ========== PHASE 3: CALCULATE RECONSTRUCTION ERRORS ==========
    print("\n" + "="*80)
    print("PHASE 3: CALCULATING RECONSTRUCTION ERRORS")
    print("="*80)
    
    # Calculate errors on test set
    test_errors, test_reconstructions = autoencoder.calculate_reconstruction_error(
        X_test, metric='mse'
    )
    
    print(f"\nTest set reconstruction errors:")
    print(f"  Min:  {test_errors.min():.6f}")
    print(f"  Max:  {test_errors.max():.6f}")
    print(f"  Mean: {test_errors.mean():.6f}")
    print(f"  Std:  {test_errors.std():.6f}")
    
    # Calculate errors on training set to determine threshold
    train_errors, _ = autoencoder.calculate_reconstruction_error(
        X_train_normal, metric='mse'
    )
    
    # Determine threshold (95th percentile of training errors)
    threshold = determine_threshold(train_errors, percentile=95)
    
    # Make predictions
    y_pred = (test_errors > threshold).astype(int)
    
    # ========== PHASE 4: VISUALIZATION & EVALUATION ==========
    print("\n" + "="*80)
    print("PHASE 4: VISUALIZATION & EVALUATION")
    print("="*80)
    
    visualizer = FraudVisualizer(output_dir='outputs')
    
    # 1. Training history
    print("\n📊 Generating training history plot...")
    visualizer.plot_training_history(history, save=True)
    
    # 2. THE MONEY SHOT: Reconstruction error distribution
    print("\n📊 Generating reconstruction error distribution (THE MONEY SHOT)...")
    visualizer.plot_reconstruction_error_distribution(
        test_errors, y_test, threshold=threshold, save=True
    )
    
    # 3. Confusion matrix
    print("\n📊 Generating confusion matrix...")
    visualizer.plot_confusion_matrix(y_test, y_pred, save=True)
    
    # 4. Precision-Recall curve
    print("\n📊 Generating Precision-Recall curve...")
    pr_auc = visualizer.plot_precision_recall_curve(y_test, test_errors, save=True)
    
    # 5. Classification report
    visualizer.print_classification_report(y_test, y_pred, threshold)
    
    # ========== PHASE 5: EXPLAINABILITY ANALYSIS ==========
    print("\n" + "="*80)
    print("PHASE 5: FEATURE-WISE EXPLAINABILITY (XAI)")
    print("="*80)
    
    # Find a fraudulent transaction that was correctly detected
    fraud_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    
    if len(fraud_indices) > 0:
        # Pick the first correctly detected fraud
        fraud_idx = fraud_indices[0]
        
        print(f"\nAnalyzing fraud transaction at index {fraud_idx}...")
        print(f"Reconstruction error: {test_errors[fraud_idx]:.6f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Error is {test_errors[fraud_idx]/threshold:.2f}x the threshold")
        
        # Get feature names
        feature_names = preprocessor.feature_columns if hasattr(preprocessor, 'feature_columns') else None
        
        # Plot explainability
        print("\n📊 Generating feature explainability plot...")
        visualizer.plot_feature_explainability(
            input_sample=X_test[fraud_idx],
            reconstructed_sample=test_reconstructions[fraud_idx],
            feature_names=feature_names,
            top_n=10,
            save=True
        )
    else:
        print("\n⚠️  No correctly detected fraud cases found. Try adjusting the threshold.")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("✨ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("  Models:")
    print("    - models/fraud_autoencoder.keras")
    print("    - models/best_autoencoder.keras")
    print("    - models/scaler.pkl")
    print("\n  Visualizations:")
    print("    - outputs/training_history.png")
    print("    - outputs/reconstruction_error_distribution.png")
    print("    - outputs/confusion_matrix.png")
    print("    - outputs/precision_recall_curve.png")
    print("    - outputs/feature_explainability.png")
    
    print("\n📊 Key Results:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Total fraud detected: {y_pred.sum()} / {y_test.sum()}")
    
    print("\n🎉 Ready for report writing!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Please download the dataset first using data_loader.py instructions.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
