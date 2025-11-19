"""
Simple Training Script - Train Models One at a Time
Avoids TensorFlow initialization issues
"""

import os
# Force CPU mode to avoid GPU/mutex issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("SIMPLE TRAINING - ONE MODEL AT A TIME")
print("="*80)

# Ask user which model to train
print("\nWhich model would you like to train?")
print("1. Standard Autoencoder (fastest, ~3-5 min)")
print("2. VAE - Variational Autoencoder (~5-7 min)")
print("3. LSTM Autoencoder (~7-10 min)")
print("4. All models (sequential, ~20 min)")

choice = input("\nEnter choice (1/2/3/4): ").strip()

# Load data first
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

from data_loader import load_data
from preprocessor import DataPreprocessor

df = load_data()
preprocessor = DataPreprocessor()
X_train_normal, X_test, y_test = preprocessor.preprocess(df)

# Save scaler
os.makedirs('models', exist_ok=True)
preprocessor.save_scaler('models/scaler.pkl')

print(f"\n✓ Data loaded: {X_train_normal.shape[0]} training samples")

# Train based on choice
if choice in ['1', '4']:
    print("\n" + "="*80)
    print("TRAINING: STANDARD AUTOENCODER")
    print("="*80)
    
    from autoencoder import FraudAutoencoder
    
    ae = FraudAutoencoder(input_dim=X_train_normal.shape[1])
    ae.build_model()
    ae.compile_model()
    ae.train(X_train_normal, epochs=50, batch_size=32, validation_split=0.1)
    ae.save_model('models/fraud_autoencoder.keras')
    
    # Calculate metrics
    train_errors = ae.calculate_reconstruction_error(X_train_normal)
    test_errors = ae.calculate_reconstruction_error(X_test)
    threshold = np.percentile(train_errors, 95)
    y_pred = (test_errors > threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n✓ Standard AE Results:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

if choice in ['2', '4']:
    print("\n" + "="*80)
    print("TRAINING: VARIATIONAL AUTOENCODER (VAE)")
    print("="*80)
    
    from vae_model import FraudVAE
    
    vae = FraudVAE(input_dim=X_train_normal.shape[1], latent_dim=7, intermediate_dims=[14])
    vae.build_model()
    vae.compile_model()
    vae.train(X_train_normal, epochs=50, batch_size=32, validation_split=0.1)
    vae.save_model('models/fraud_vae.keras')
    
    # Calculate metrics
    train_errors, _, _ = vae.calculate_reconstruction_error(X_train_normal, n_samples=10)
    test_errors, test_unc, _ = vae.calculate_reconstruction_error(X_test, n_samples=10)
    threshold = np.percentile(train_errors, 95)
    y_pred = (test_errors > threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n✓ VAE Results:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Mean Uncertainty: {np.mean(test_unc):.4f}")

if choice in ['3', '4']:
    print("\n" + "="*80)
    print("TRAINING: LSTM AUTOENCODER")
    print("="*80)
    
    from lstm_autoencoder import LSTMAutoencoder
    
    lstm_ae = LSTMAutoencoder(
        input_dim=X_train_normal.shape[1],
        sequence_length=10,
        latent_dim=7
    )
    lstm_ae.build_model()
    lstm_ae.compile_model()
    lstm_ae.train(X_train_normal, epochs=50, batch_size=32)
    lstm_ae.save_model('models/lstm_autoencoder.keras')
    
    # Calculate metrics
    train_errors, _ = lstm_ae.calculate_reconstruction_error(X_train_normal)
    test_errors, _ = lstm_ae.calculate_reconstruction_error(X_test)
    threshold = np.percentile(train_errors, 95)
    
    # Handle sequence length for predictions
    y_pred_full = np.zeros(len(y_test))
    y_pred_full[:len(test_errors)] = (test_errors > threshold).astype(int)
    if len(test_errors) < len(y_test):
        y_pred_full[len(test_errors):] = y_pred_full[len(test_errors)-1]
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred_full, zero_division=0)
    recall = recall_score(y_test, y_pred_full)
    f1 = f1_score(y_test, y_pred_full, zero_division=0)
    
    print(f"\n✓ LSTM-AE Results:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nNext step: Launch web app")
print("  streamlit run app.py")
