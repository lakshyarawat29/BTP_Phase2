"""
Variational Autoencoder (VAE) for Fraud Detection
Advanced model with probabilistic latent space
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, ops
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

class Sampling(layers.Layer):
    """
    Custom layer for sampling from the latent distribution
    Uses the reparameterization trick: z = mean + std * epsilon
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(layers.Layer):
    """
    Custom layer to compute KL divergence loss
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Use Keras ops instead of TensorFlow ops
        kl_loss = -0.5 * ops.mean(
            z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return z_mean  # Pass through

class FraudVAE:
    """
    Variational Autoencoder for fraud detection
    Provides probabilistic anomaly detection with uncertainty estimates
    """
    
    def __init__(self, input_dim, latent_dim=7, intermediate_dims=[14]):
        """
        Initialize VAE
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space (bottleneck)
            intermediate_dims: List of intermediate layer dimensions
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.history = None
        
    def build_encoder(self):
        """Build the encoder network"""
        encoder_inputs = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = encoder_inputs
        
        # Encoder layers
        for i, dim in enumerate(self.intermediate_dims):
            x = layers.Dense(dim, activation='relu', name=f'encoder_{i+1}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sample from latent distribution
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """Build the decoder network"""
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        x = latent_inputs
        
        # Decoder layers (mirror of encoder)
        for i, dim in enumerate(reversed(self.intermediate_dims)):
            x = layers.Dense(dim, activation='relu', name=f'decoder_{i+1}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        decoder_outputs = layers.Dense(self.input_dim, activation='linear', name='decoder_output')(x)
        
        self.decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
        return self.decoder
    
    def build_model(self):
        """Build complete VAE model"""
        print("="*80)
        print("BUILDING VARIATIONAL AUTOENCODER (VAE)")
        print("="*80)
        
        # Build encoder and decoder
        self.build_encoder()
        self.build_decoder()
        
        # Connect encoder and decoder
        encoder_inputs = layers.Input(shape=(self.input_dim,), name='vae_input')
        z_mean, z_log_var, z = self.encoder(encoder_inputs)
        
        # Add KL divergence loss through custom layer
        _ = KLDivergenceLayer()([z_mean, z_log_var])
        
        reconstructed = self.decoder(z)
        
        # Create VAE model
        self.vae = models.Model(encoder_inputs, reconstructed, name='vae')
        
        print(f"\n✓ VAE Architecture:")
        print(f"   Input: {self.input_dim} features")
        print(f"   Encoder: {' → '.join(map(str, self.intermediate_dims))}")
        print(f"   Latent Space: {self.latent_dim} dimensions (probabilistic)")
        print(f"   Decoder: {' → '.join(map(str, reversed(self.intermediate_dims)))}")
        print(f"   Output: {self.input_dim} features")
        print(f"\n✓ VAE uses KL divergence for regularization")
        
        return self.vae
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the VAE model"""
        if self.vae is None:
            raise ValueError("Model must be built before compiling")
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.vae.compile(
            optimizer=opt,
            loss='mse',
            metrics=['mae', 'mse']
        )
        print(f"\n✓ VAE compiled with optimizer={optimizer}, lr={learning_rate}")
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=32,
              validation_split=0.1, verbose=1):
        """Train the VAE"""
        print(f"\n{'='*80}")
        print("TRAINING VARIATIONAL AUTOENCODER")
        print(f"{'='*80}")
        
        print(f"\nTraining samples: {X_train.shape[0]}")
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
                'models/best_vae.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        os.makedirs('models', exist_ok=True)
        
        # Train
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"\n✓ Training completed!")
        return self.history
    
    def calculate_reconstruction_error(self, X, n_samples=10):
        """
        Calculate reconstruction error with uncertainty estimation
        
        Args:
            X: Input data
            n_samples: Number of samples from latent distribution for uncertainty
            
        Returns:
            errors: Mean reconstruction errors
            uncertainties: Standard deviation of errors (epistemic uncertainty)
            reconstructions: Mean reconstructions
        """
        all_reconstructions = []
        
        # Sample multiple reconstructions
        for _ in range(n_samples):
            recon = self.vae.predict(X, verbose=0)
            all_reconstructions.append(recon)
        
        all_reconstructions = np.array(all_reconstructions)
        
        # Mean reconstruction
        mean_reconstructions = np.mean(all_reconstructions, axis=0)
        
        # Calculate errors for each sample
        all_errors = []
        for recon in all_reconstructions:
            errors = np.mean(np.power(X - recon, 2), axis=1)
            all_errors.append(errors)
        
        all_errors = np.array(all_errors)
        
        # Mean error and uncertainty (std of errors)
        mean_errors = np.mean(all_errors, axis=0)
        uncertainties = np.std(all_errors, axis=0)
        
        return mean_errors, uncertainties, mean_reconstructions
    
    def get_latent_representation(self, X):
        """Get mean and variance of latent representation"""
        z_mean, z_log_var, _ = self.encoder.predict(X, verbose=0)
        return z_mean, z_log_var
    
    def save_model(self, filepath='models/fraud_vae.keras'):
        """Save the trained VAE"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.vae.save(filepath)
        self.encoder.save(filepath.replace('.keras', '_encoder.keras'))
        self.decoder.save(filepath.replace('.keras', '_decoder.keras'))
        print(f"\n✓ VAE saved to {filepath}")
    
    def load_model(self, filepath='models/fraud_vae.keras'):
        """Load a trained VAE"""
        custom_objects = {
            'Sampling': Sampling,
            'KLDivergenceLayer': KLDivergenceLayer
        }
        self.vae = keras.models.load_model(filepath, custom_objects=custom_objects)
        self.encoder = keras.models.load_model(
            filepath.replace('.keras', '_encoder.keras'),
            custom_objects=custom_objects
        )
        self.decoder = keras.models.load_model(filepath.replace('.keras', '_decoder.keras'))
        print(f"✓ VAE loaded from {filepath}")

if __name__ == "__main__":
    print("\nTesting VAE architecture...")
    
    # Test with dummy data
    input_dim = 29
    vae = FraudVAE(input_dim=input_dim, latent_dim=7, intermediate_dims=[14])
    vae.build_model()
    vae.compile_model()
    
    print("\n✨ VAE architecture built successfully!")
    print("\nVAE Advantages:")
    print("  • Probabilistic latent space (smoother than standard autoencoder)")
    print("  • Provides uncertainty estimates for predictions")
    print("  • Better generalization through KL regularization")
    print("  • Can generate synthetic fraud-like patterns")
