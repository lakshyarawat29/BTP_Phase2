"""
LSTM Autoencoder for Fraud Detection
Captures temporal patterns in transaction sequences
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

class LSTMAutoencoder:
    """
    LSTM-based Autoencoder for sequence anomaly detection
    Captures temporal patterns in transaction data
    """
    
    def __init__(self, input_dim, sequence_length=10, latent_dim=7):
        """
        Initialize LSTM Autoencoder
        
        Args:
            input_dim: Number of features per timestep
            sequence_length: Number of timesteps in sequence
            latent_dim: Dimension of latent representation
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
    
    def build_model(self):
        """Build LSTM Autoencoder architecture"""
        print("="*80)
        print("BUILDING LSTM AUTOENCODER")
        print("="*80)
        
        # Encoder
        encoder_inputs = layers.Input(
            shape=(self.sequence_length, self.input_dim),
            name='encoder_input'
        )
        
        # LSTM Encoder layers
        x = layers.LSTM(64, activation='tanh', return_sequences=True, name='lstm_1')(encoder_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, activation='tanh', return_sequences=True, name='lstm_2')(x)
        x = layers.Dropout(0.2)(x)
        
        # Bottleneck (encode to latent representation)
        encoded = layers.LSTM(self.latent_dim, activation='tanh', name='bottleneck')(x)
        
        # Create encoder model
        self.encoder = models.Model(encoder_inputs, encoded, name='lstm_encoder')
        
        # Decoder - Repeat vector to match sequence length
        x = layers.RepeatVector(self.sequence_length)(encoded)
        
        # LSTM Decoder layers
        x = layers.LSTM(self.latent_dim, activation='tanh', return_sequences=True, name='lstm_decode_1')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, activation='tanh', return_sequences=True, name='lstm_decode_2')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64, activation='tanh', return_sequences=True, name='lstm_decode_3')(x)
        
        # Output layer - reconstruct input
        decoded = layers.TimeDistributed(
            layers.Dense(self.input_dim, activation='linear'),
            name='output'
        )(x)
        
        # Complete autoencoder
        self.model = models.Model(encoder_inputs, decoded, name='lstm_autoencoder')
        
        print(f"\n✓ LSTM Autoencoder Architecture:")
        print(f"   Input: ({self.sequence_length} timesteps, {self.input_dim} features)")
        print(f"   Encoder: LSTM(64) → LSTM(32) → LSTM({self.latent_dim})")
        print(f"   Latent: {self.latent_dim} dimensions")
        print(f"   Decoder: LSTM({self.latent_dim}) → LSTM(32) → LSTM(64)")
        print(f"   Output: ({self.sequence_length} timesteps, {self.input_dim} features)")
        
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=opt,
            loss='mse',
            metrics=['mae']
        )
        print(f"\n✓ LSTM Autoencoder compiled with optimizer={optimizer}, lr={learning_rate}")
    
    def prepare_sequences(self, data, sequence_length=None):
        """
        Convert flat data into sequences for LSTM
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            sequence_length: length of sequences (uses self.sequence_length if None)
            
        Returns:
            sequences: numpy array of shape (n_sequences, sequence_length, n_features)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        # Create overlapping sequences
        sequences = []
        for i in range(n_samples - sequence_length + 1):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"\n✓ Created {len(sequences)} sequences from {n_samples} samples")
        print(f"   Sequence shape: ({sequence_length}, {n_features})")
        
        return sequences
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=32, verbose=1):
        """Train the LSTM Autoencoder"""
        print(f"\n{'='*80}")
        print("TRAINING LSTM AUTOENCODER ON TRANSACTION SEQUENCES")
        print(f"{'='*80}")
        
        # Prepare sequences
        print("\nPreparing training sequences...")
        X_train_seq = self.prepare_sequences(X_train)
        
        if X_val is not None:
            print("Preparing validation sequences...")
            X_val_seq = self.prepare_sequences(X_val)
            validation_data = (X_val_seq, X_val_seq)
        else:
            validation_data = None
        
        print(f"\nTraining sequences: {X_train_seq.shape[0]}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_lstm_autoencoder.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        os.makedirs('models', exist_ok=True)
        
        # Train
        self.history = self.model.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.1 if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def calculate_reconstruction_error(self, X):
        """
        Calculate reconstruction error for sequences
        
        Args:
            X: Input data (flat or sequences)
            
        Returns:
            errors: Reconstruction error per sequence
            reconstructions: Reconstructed sequences
        """
        # Prepare sequences if input is flat
        if len(X.shape) == 2:
            X_seq = self.prepare_sequences(X)
        else:
            X_seq = X
        
        # Reconstruct
        reconstructions = self.model.predict(X_seq, verbose=0)
        
        # Calculate MSE per sequence (average across timesteps and features)
        errors = np.mean(np.mean(np.power(X_seq - reconstructions, 2), axis=2), axis=1)
        
        return errors, reconstructions
    
    def get_latent_representation(self, X):
        """Get latent representation from encoder"""
        if len(X.shape) == 2:
            X_seq = self.prepare_sequences(X)
        else:
            X_seq = X
        
        return self.encoder.predict(X_seq, verbose=0)
    
    def save_model(self, filepath='models/lstm_autoencoder.keras'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        self.encoder.save(filepath.replace('.keras', '_encoder.keras'))
        print(f"\n✓ LSTM Autoencoder saved to {filepath}")
    
    def load_model(self, filepath='models/lstm_autoencoder.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.encoder = keras.models.load_model(filepath.replace('.keras', '_encoder.keras'))
        print(f"✓ LSTM Autoencoder loaded from {filepath}")

if __name__ == "__main__":
    print("\nTesting LSTM Autoencoder architecture...")
    
    # Test with dummy data
    input_dim = 29
    sequence_length = 10
    
    lstm_ae = LSTMAutoencoder(
        input_dim=input_dim,
        sequence_length=sequence_length,
        latent_dim=7
    )
    lstm_ae.build_model()
    lstm_ae.compile_model()
    
    # Test sequence preparation
    dummy_data = np.random.randn(100, input_dim)
    sequences = lstm_ae.prepare_sequences(dummy_data)
    
    print("\n✨ LSTM Autoencoder built successfully!")
    print("\nLSTM Advantages:")
    print("  • Captures temporal patterns in transaction sequences")
    print("  • Detects anomalies based on transaction history")
    print("  • Better for time-series fraud patterns")
    print("  • Can detect unusual sequences even if individual transactions seem normal")
