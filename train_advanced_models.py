"""
Advanced Model Comparison and Training
Compares Standard Autoencoder, VAE, and LSTM Autoencoder
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import all models
from autoencoder import FraudAutoencoder
from vae_model import FraudVAE
from lstm_autoencoder import LSTMAutoencoder
from data_loader import load_data
from preprocessor import DataPreprocessor

class AdvancedModelComparison:
    """Compare Standard AE, VAE, and LSTM-AE on fraud detection"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.errors = {}
        self.thresholds = {}
        
    def load_and_prepare_data(self):
        """Load and preprocess data"""
        print("\n" + "="*80)
        print("LOADING AND PREPARING DATA FOR ADVANCED MODELS")
        print("="*80)
        
        # Load data
        df = load_data()
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train_normal, X_test, y_test, scaler = preprocessor.preprocess(df)
        
        # Save scaler
        with open('models/scaler_advanced.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n✓ Data loaded and preprocessed")
        print(f"   Training samples (normal only): {X_train_normal.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train_normal.shape[1]}")
        
        return X_train_normal, X_test, y_test
    
    def train_standard_ae(self, X_train, X_test, y_test):
        """Train standard autoencoder"""
        print("\n" + "="*80)
        print("MODEL 1: STANDARD AUTOENCODER (Baseline)")
        print("="*80)
        
        start_time = time.time()
        
        # Build and train
        ae = FraudAutoencoder(input_dim=X_train.shape[1])
        ae.build_model()
        ae.compile_model()
        ae.train(X_train, epochs=50, batch_size=32, validation_split=0.1)
        ae.save_model('models/comparison_standard_ae.keras')
        
        train_time = time.time() - start_time
        
        # Calculate errors
        train_errors = ae.calculate_reconstruction_error(X_train)
        test_errors = ae.calculate_reconstruction_error(X_test)
        
        # Set threshold (95th percentile)
        threshold = np.percentile(train_errors, 95)
        
        # Predictions
        y_pred = (test_errors > threshold).astype(int)
        
        # Metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.models['Standard AE'] = ae
        self.errors['Standard AE'] = (train_errors, test_errors)
        self.thresholds['Standard AE'] = threshold
        self.results['Standard AE'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'predictions': y_pred
        }
        
        print(f"\n✓ Standard AE Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
    def train_vae(self, X_train, X_test, y_test):
        """Train Variational Autoencoder"""
        print("\n" + "="*80)
        print("MODEL 2: VARIATIONAL AUTOENCODER (VAE)")
        print("="*80)
        
        start_time = time.time()
        
        # Build and train
        vae = FraudVAE(input_dim=X_train.shape[1], latent_dim=7, intermediate_dims=[14])
        vae.build_model()
        vae.compile_model()
        vae.train(X_train, epochs=50, batch_size=32, validation_split=0.1)
        vae.save_model('models/comparison_vae.keras')
        
        train_time = time.time() - start_time
        
        # Calculate errors with uncertainty
        train_errors, train_unc, _ = vae.calculate_reconstruction_error(X_train, n_samples=10)
        test_errors, test_unc, _ = vae.calculate_reconstruction_error(X_test, n_samples=10)
        
        # Set threshold
        threshold = np.percentile(train_errors, 95)
        
        # Predictions
        y_pred = (test_errors > threshold).astype(int)
        
        # Metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.models['VAE'] = vae
        self.errors['VAE'] = (train_errors, test_errors, test_unc)
        self.thresholds['VAE'] = threshold
        self.results['VAE'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'predictions': y_pred,
            'uncertainty': test_unc
        }
        
        print(f"\n✓ VAE Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Mean uncertainty: {np.mean(test_unc):.4f}")
        
    def train_lstm_ae(self, X_train, X_test, y_test):
        """Train LSTM Autoencoder"""
        print("\n" + "="*80)
        print("MODEL 3: LSTM AUTOENCODER")
        print("="*80)
        
        start_time = time.time()
        
        # Build and train
        lstm_ae = LSTMAutoencoder(
            input_dim=X_train.shape[1],
            sequence_length=10,
            latent_dim=7
        )
        lstm_ae.build_model()
        lstm_ae.compile_model()
        lstm_ae.train(X_train, epochs=50, batch_size=32)
        lstm_ae.save_model('models/comparison_lstm_ae.keras')
        
        train_time = time.time() - start_time
        
        # Calculate errors
        train_errors, _ = lstm_ae.calculate_reconstruction_error(X_train)
        test_errors, _ = lstm_ae.calculate_reconstruction_error(X_test)
        
        # Set threshold
        threshold = np.percentile(train_errors, 95)
        
        # Predictions - need to handle sequence length
        # For each test sample, we predict based on its sequence
        y_pred_full = np.zeros(len(y_test))
        y_pred_full[:len(test_errors)] = (test_errors > threshold).astype(int)
        
        # For remaining samples (within sequence_length of end), use last prediction
        if len(test_errors) < len(y_test):
            y_pred_full[len(test_errors):] = y_pred_full[len(test_errors)-1]
        
        # Metrics
        precision = precision_score(y_test, y_pred_full, zero_division=0)
        recall = recall_score(y_test, y_pred_full)
        f1 = f1_score(y_test, y_pred_full, zero_division=0)
        
        self.models['LSTM-AE'] = lstm_ae
        self.errors['LSTM-AE'] = (train_errors, test_errors)
        self.thresholds['LSTM-AE'] = threshold
        self.results['LSTM-AE'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time,
            'predictions': y_pred_full
        }
        
        print(f"\n✓ LSTM-AE Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
    
    def print_comparison_table(self):
        """Print comparison table"""
        print("\n" + "="*80)
        print("ADVANCED MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Training Time (s)': f"{metrics['train_time']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # Find best models
        print("\n" + "="*80)
        print("BEST PERFORMERS")
        print("="*80)
        
        best_precision = max(self.results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1'])
        
        print(f"\n✨ Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
        print(f"✨ Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        print(f"✨ Best F1-Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    
    def plot_comparison(self):
        """Create comprehensive comparison visualizations"""
        print("\n📊 Generating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance metrics comparison
        ax = axes[0, 0]
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.25
        
        precisions = [self.results[m]['precision'] for m in models]
        recalls = [self.results[m]['recall'] for m in models]
        f1s = [self.results[m]['f1'] for m in models]
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Training time comparison
        ax = axes[0, 1]
        train_times = [self.results[m]['train_time'] for m in models]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax.barh(models, train_times, color=colors, alpha=0.7)
        ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Error distribution comparison
        ax = axes[1, 0]
        for i, model_name in enumerate(models):
            _, test_errors = self.errors[model_name][:2]
            ax.hist(test_errors, bins=50, alpha=0.5, label=model_name, density=True)
        
        ax.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 4. Confusion matrices
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create mini confusion matrices
        for i, model_name in enumerate(models):
            from sklearn.metrics import confusion_matrix
            
            # Get predictions
            y_pred = self.results[model_name]['predictions']
            
            # We need y_test - get it from the first model's context
            # For visualization purposes, create a small CM display
            ax_sub = fig.add_subplot(2, 2, 4)
            ax_sub.text(0.5, 0.7 - i*0.25, f"{model_name}:", 
                       fontsize=11, fontweight='bold', ha='center')
            ax_sub.text(0.5, 0.6 - i*0.25,
                       f"Precision: {self.results[model_name]['precision']:.3f} | "
                       f"Recall: {self.results[model_name]['recall']:.3f}",
                       fontsize=10, ha='center')
            ax_sub.axis('off')
        
        ax_sub.set_title('Summary Metrics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('outputs/advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: outputs/advanced_model_comparison.png")
    
    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        print("\n" + "🚀 "*20)
        print("ADVANCED FRAUD DETECTION MODEL COMPARISON")
        print("🚀 "*20)
        
        # Load data
        X_train, X_test, y_test = self.load_and_prepare_data()
        
        # Train all models
        self.train_standard_ae(X_train, X_test, y_test)
        self.train_vae(X_train, X_test, y_test)
        self.train_lstm_ae(X_train, X_test, y_test)
        
        # Print comparison
        self.print_comparison_table()
        
        # Plot comparison
        self.plot_comparison()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE!")
        print("="*80)
        print("\n✨ Key Insights:")
        print("   • VAE provides uncertainty estimates (helpful for confidence scoring)")
        print("   • LSTM-AE captures temporal patterns (good for sequence anomalies)")
        print("   • Standard AE is fastest and simplest (good baseline)")
        print("\nAll models saved to models/ directory")
        print("Comparison visualization saved to outputs/")

if __name__ == "__main__":
    comparison = AdvancedModelComparison()
    comparison.run_full_comparison()
