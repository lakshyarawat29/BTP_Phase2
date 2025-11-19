"""
Interactive Fraud Detection Demo
Test the model with custom transactions
"""

import numpy as np
from autoencoder import FraudAutoencoder
from preprocessor import DataPreprocessor
import pickle

def load_model_and_scaler():
    """Load trained model and scaler"""
    autoencoder = FraudAutoencoder(input_dim=29)
    autoencoder.load_model('models/fraud_autoencoder.keras')
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Calculate threshold from training data
    from data_loader import load_data
    df = load_data('data/creditcard.csv')
    preprocessor = DataPreprocessor(use_robust_scaler=True)
    X_train_normal, _, _ = preprocessor.preprocess(df, drop_time=False)
    train_errors, _ = autoencoder.calculate_reconstruction_error(X_train_normal)
    threshold = np.percentile(train_errors, 95)
    
    return autoencoder, scaler, threshold

def analyze_transaction(transaction, autoencoder, threshold, feature_names):
    """
    Analyze a single transaction
    
    Args:
        transaction: numpy array of features
        autoencoder: trained model
        threshold: detection threshold
        feature_names: list of feature names
    """
    # Calculate reconstruction error
    error, reconstruction = autoencoder.calculate_reconstruction_error(
        transaction.reshape(1, -1)
    )
    error = error[0]
    reconstruction = reconstruction[0]
    
    # Determine if fraud
    is_fraud = error > threshold
    confidence = (error / threshold) if is_fraud else (1 - error / threshold)
    
    # Feature-wise errors
    feature_errors = np.abs(transaction - reconstruction)
    top_5_indices = np.argsort(feature_errors)[-5:][::-1]
    
    print("\n" + "="*80)
    print("TRANSACTION ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Reconstruction Error: {error:.6f}")
    print(f"🎯 Threshold: {threshold:.6f}")
    print(f"📈 Error / Threshold Ratio: {error/threshold:.2f}x")
    
    if is_fraud:
        print(f"\n🚨 FRAUD DETECTED! (Confidence: {confidence*100:.1f}%)")
    else:
        print(f"\n✅ NORMAL TRANSACTION (Confidence: {confidence*100:.1f}%)")
    
    print(f"\n🔍 Top 5 Suspicious Features:")
    for i, idx in enumerate(top_5_indices, 1):
        feature_name = feature_names[idx] if feature_names else f"Feature_{idx}"
        print(f"   {i}. {feature_name}: Error = {feature_errors[idx]:.4f}")
    
    return is_fraud, error

def demo_mode():
    """Interactive demo mode"""
    print("\n" + "="*80)
    print("FRAUD DETECTION INTERACTIVE DEMO")
    print("="*80)
    
    print("\nLoading model...")
    autoencoder, scaler, threshold = load_model_and_scaler()
    
    # Get feature names (V1-V28, Time, Amount)
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    
    print("\n✅ Model loaded successfully!")
    print(f"Detection threshold: {threshold:.6f}")
    
    # Load sample data for testing
    from data_loader import load_data
    df = load_data('data/creditcard.csv')
    
    while True:
        print("\n" + "="*80)
        print("OPTIONS:")
        print("1. Test a random NORMAL transaction")
        print("2. Test a random FRAUD transaction")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '3':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            # Random normal transaction
            normal_df = df[df['Class'] == 0]
            sample = normal_df.sample(1).drop('Class', axis=1).values[0]
            
            print("\n📤 Testing NORMAL transaction...")
            analyze_transaction(sample, autoencoder, threshold, feature_names)
            
        elif choice == '2':
            # Random fraud transaction
            fraud_df = df[df['Class'] == 1]
            if len(fraud_df) == 0:
                print("\n⚠️  No fraud transactions available!")
                continue
            
            sample = fraud_df.sample(1).drop('Class', axis=1).values[0]
            
            print("\n📤 Testing FRAUD transaction...")
            analyze_transaction(sample, autoencoder, threshold, feature_names)
        else:
            print("\n❌ Invalid choice!")

if __name__ == "__main__":
    try:
        demo_mode()
    except FileNotFoundError:
        print("\n❌ Error: Model files not found!")
        print("Please run train.py first to train the model.")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
