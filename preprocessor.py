"""
Data Preprocessing Module for Fraud Detection Autoencoder
Implements the "Critical Split" strategy for unsupervised learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os

class DataPreprocessor:
    """
    Handles data preprocessing for unsupervised fraud detection
    """
    
    def __init__(self, use_robust_scaler=True):
        """
        Initialize preprocessor
        
        Args:
            use_robust_scaler (bool): If True, uses RobustScaler (better for outliers)
        """
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.feature_columns = None
        
    def preprocess(self, df, drop_time=False, test_size=0.2, random_state=42):
        """
        Performs the critical split for unsupervised learning
        
        Args:
            df (pd.DataFrame): Raw credit card dataset
            drop_time (bool): Whether to drop the Time column
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train_normal, X_test, y_test, scaler)
        """
        print("=" * 80)
        print("PHASE 1: DATA PREPROCESSING - THE CRITICAL SPLIT")
        print("=" * 80)
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Step 1: Handle Time column
        if drop_time and 'Time' in data.columns:
            print("\n✓ Dropping 'Time' column")
            data = data.drop('Time', axis=1)
        elif 'Time' in data.columns:
            print("\n✓ Keeping 'Time' column (will be scaled)")
        
        # Step 2: Separate features and labels
        if 'Class' not in data.columns:
            raise ValueError("Dataset must contain 'Class' column")
        
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        print(f"\nOriginal dataset shape: {X.shape}")
        print(f"Features: {X.shape[1]}")
        
        # Step 3: Scale the Amount column (and Time if present)
        cols_to_scale = ['Amount']
        if 'Time' in X.columns:
            cols_to_scale.append('Time')
        
        print(f"\nScaling columns: {cols_to_scale}")
        X[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
        
        self.feature_columns = X.columns.tolist()
        
        # Step 4: Train-Test Split
        print(f"\n{'='*80}")
        print("PERFORMING THE CRITICAL SPLIT")
        print(f"{'='*80}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Step 5: THE CRITICAL STEP - Filter training data to ONLY normal transactions
        print(f"\n{'='*80}")
        print("🔑 CRITICAL STEP: Filtering training set to NORMAL transactions only")
        print(f"{'='*80}")
        
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask].copy()
        
        print(f"\nBefore filtering: {len(X_train)} transactions")
        print(f"After filtering: {len(X_train_normal)} NORMAL transactions")
        print(f"Removed: {len(X_train) - len(X_train_normal)} fraudulent transactions")
        
        # Step 6: Convert to numpy arrays
        X_train_normal = X_train_normal.values
        X_test = X_test.values
        y_test = y_test.values
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"\nTraining set (NORMAL only): {X_train_normal.shape}")
        print(f"Test set (Mixed): {X_test.shape}")
        print(f"\nTest set class distribution:")
        print(f"  Normal (0): {(y_test == 0).sum()} samples")
        print(f"  Fraud (1): {(y_test == 1).sum()} samples")
        print(f"  Fraud percentage: {(y_test.sum() / len(y_test) * 100):.4f}%")
        
        return X_train_normal, X_test, y_test
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        """Load a previously fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")

if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_data
    
    try:
        print("\nTesting DataPreprocessor...")
        df = load_data()
        
        preprocessor = DataPreprocessor(use_robust_scaler=True)
        X_train, X_test, y_test = preprocessor.preprocess(df, drop_time=False)
        
        print("\n✨ Preprocessing completed successfully!")
        
    except FileNotFoundError:
        print("\n⚠️  Please download the dataset first.")
