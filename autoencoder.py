"""
Deep Autoencoder Architecture for Fraud Detection
Implements the bottleneck architecture for anomaly detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

class FraudAutoencoder:
    """
    Deep Autoencoder for unsupervised fraud detection
    """
    
    def __init__(self, input_dim, encoding_dims=[14, 7]):
        """
        Initialize the autoencoder architecture
        
        Args:
            input_dim (int): Number of input features
            encoding_dims (list): Dimensions for encoder layers (bottleneck is last)
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the autoencoder architecture
        
        Architecture:
            Input (29) -> Encoder (14) -> Bottleneck (7) -> Decoder (14) -> Output (29)
        """
        print("=" * 80)
        print("BUILDING AUTOENCODER ARCHITECTURE")
        print("=" * 80)
        
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,), name='input')
        
        # Encoder
        print(f"\n📥 Input Layer: {self.input_dim} features")
        encoded = input_layer
        
        for i, dim in enumerate(self.encoding_dims):
            encoded = layers.Dense(
                dim, 
                activation='relu',
                name=f'encoder_{i+1}'
            )(encoded)
            print(f"   ↓ Encoder Layer {i+1}: {dim} neurons (ReLU)")
        
        print(f"\n🔒 Bottleneck Layer: {self.encoding_dims[-1]} neurons")
        print("   (This is the compressed 'normal transaction' manifold)")
        
        # Decoder (mirror of encoder)
        decoded = encoded
        
        for i, dim in enumerate(reversed(self.encoding_dims[:-1])):
            decoded = layers.Dense(
                dim,
                activation='relu',
                name=f'decoder_{i+1}'
            )(decoded)
            print(f"   ↑ Decoder Layer {i+1}: {dim} neurons (ReLU)")
        
        # Output layer
        output_layer = layers.Dense(
            self.input_dim,
            activation='linear',  # Can use 'sigmoid' if data is normalized to [0,1]
            name='output'
        )(decoded)
        
        print(f"   ↑ Output Layer: {self.input_dim} features (Reconstruction)")
        
        # Create model
        self.model = models.Model(inputs=input_layer, outputs=output_layer, name='fraud_autoencoder')
        
        print(f"\n{'='*80}")
        print("MODEL SUMMARY")
        print(f"{'='*80}")
        self.model.summary()
        
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001, loss='mse'):
        """
        Compile the model
        
        Args:
            optimizer (str): Optimizer name
            learning_rate (float): Learning rate
            loss (str): Loss function ('mse' or 'mae')
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling. Call build_model() first.")
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate) if optimizer == 'adam' else optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['mae', 'mse']
        )
        
        print(f"\n✓ Model compiled with optimizer={optimizer}, loss={loss}, lr={learning_rate}")
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=32, 
              validation_split=0.1, callbacks=None, verbose=1):
        """
        Train the autoencoder
        
        Args:
            X_train (np.array): Training data (NORMAL transactions only)
            X_val (np.array): Validation data (optional, can be mixed)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split if X_val not provided
            callbacks (list): Keras callbacks
            verbose (int): Verbosity level
        """
        print(f"\n{'='*80}")
        print("TRAINING AUTOENCODER ON NORMAL TRANSACTIONS")
        print(f"{'='*80}")
        
        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/best_autoencoder.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train (input = output for autoencoders)
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, X_train,  # Input and target are the same
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def calculate_reconstruction_error(self, X, metric='mse'):
        """
        Calculate reconstruction error for each sample
        
        Args:
            X (np.array): Input data
            metric (str): 'mse' or 'mae'
            
        Returns:
            np.array: Reconstruction error for each sample
        """
        reconstructions = self.model.predict(X, verbose=0)
        
        if metric == 'mse':
            errors = np.mean(np.power(X - reconstructions, 2), axis=1)
        elif metric == 'mae':
            errors = np.mean(np.abs(X - reconstructions), axis=1)
        else:
            raise ValueError("metric must be 'mse' or 'mae'")
        
        return errors, reconstructions
    
    def save_model(self, filepath='models/fraud_autoencoder.keras'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/fraud_autoencoder.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")

if __name__ == "__main__":
    print("\nTesting FraudAutoencoder architecture...")
    
    # Test with dummy data
    input_dim = 29  # For credit card dataset
    autoencoder = FraudAutoencoder(input_dim=input_dim, encoding_dims=[14, 7])
    autoencoder.build_model()
    autoencoder.compile_model()
    
    print("\n✨ Autoencoder architecture built successfully!")
