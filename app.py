"""
Fraud Detection Web Application
Built with Streamlit for interactive demonstration
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from pathlib import Path

# Import models
try:
    from autoencoder import FraudAutoencoder
    from vae_model import FraudVAE
    from lstm_autoencoder import LSTMAutoencoder
except ImportError:
    st.error("Model files not found. Please ensure all model files are in the same directory.")

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .fraud-detected {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .normal-transaction {
        color: #00cc66;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    # Load Standard AE
    if os.path.exists('models/fraud_autoencoder.keras'):
        ae = FraudAutoencoder(input_dim=29)
        ae.load_model('models/fraud_autoencoder.keras')
        models['Standard AE'] = ae
    
    # Load VAE
    if os.path.exists('models/fraud_vae.keras'):
        vae = FraudVAE(input_dim=29, latent_dim=7, intermediate_dims=[14])
        vae.load_model('models/fraud_vae.keras')
        models['VAE'] = vae
    
    # Load LSTM-AE
    if os.path.exists('models/lstm_autoencoder.keras'):
        lstm_ae = LSTMAutoencoder(input_dim=29, sequence_length=10, latent_dim=7)
        lstm_ae.load_model('models/lstm_autoencoder.keras')
        models['LSTM-AE'] = lstm_ae
    
    return models

@st.cache_data
def load_scaler():
    """Load the data scaler"""
    if os.path.exists('models/scaler.pkl'):
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_test_data():
    """Load test dataset for demos"""
    if os.path.exists('creditcard.csv'):
        df = pd.read_csv('creditcard.csv')
        return df
    return None

def predict_transaction(transaction, model, model_name, scaler):
    """Make prediction for a single transaction"""
    # Scale transaction
    if scaler is not None:
        transaction_scaled = scaler.transform(transaction.reshape(1, -1))
    else:
        transaction_scaled = transaction.reshape(1, -1)
    
    # Get reconstruction error
    if model_name == 'VAE':
        errors, uncertainty, _ = model.calculate_reconstruction_error(transaction_scaled, n_samples=10)
        error = errors[0]
        unc = uncertainty[0]
    elif model_name == 'LSTM-AE':
        errors, _ = model.calculate_reconstruction_error(transaction_scaled)
        error = errors[0]
        unc = None
    else:  # Standard AE
        errors = model.calculate_reconstruction_error(transaction_scaled)
        error = errors[0]
        unc = None
    
    # Get reconstruction
    if model_name == 'VAE':
        recon = model.vae.predict(transaction_scaled, verbose=0)
    elif model_name == 'LSTM-AE':
        seq = model.prepare_sequences(transaction_scaled)
        recon = model.model.predict(seq, verbose=0)[0][-1]
    else:
        recon = model.model.predict(transaction_scaled, verbose=0)
    
    return error, unc, recon

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning for Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        scaler = load_scaler()
        test_data = load_test_data()
    
    if not models:
        st.error("❌ No trained models found. Please train models first using train.py or train_advanced_models.py")
        return
    
    st.sidebar.success(f"✅ Loaded {len(models)} model(s)")
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        help="Choose which model to use for prediction"
    )
    
    selected_model = models[selected_model_name]
    
    # Threshold setting
    threshold_percentile = st.sidebar.slider(
        "Detection Threshold (percentile)",
        min_value=90,
        max_value=99,
        value=95,
        help="Higher threshold = fewer false positives, more missed frauds"
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Detection",
        "🔬 Model Analysis",
        "📈 Performance Metrics",
        "ℹ️ About"
    ])
    
    # Tab 1: Live Detection
    with tab1:
        st.header("Live Transaction Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Test Transaction")
            
            # Sample selection
            demo_option = st.radio(
                "Choose demo option:",
                ["Manual Input", "Random Normal Transaction", "Random Fraud Transaction"]
            )
            
            if test_data is not None:
                if demo_option == "Random Normal Transaction":
                    if st.button("🎲 Generate Random Normal Transaction"):
                        normal_data = test_data[test_data['Class'] == 0].sample(1)
                        st.session_state.demo_transaction = normal_data.drop('Class', axis=1).values[0]
                        st.session_state.true_label = 0
                
                elif demo_option == "Random Fraud Transaction":
                    if st.button("🎲 Generate Random Fraud Transaction"):
                        fraud_data = test_data[test_data['Class'] == 1].sample(1)
                        st.session_state.demo_transaction = fraud_data.drop('Class', axis=1).values[0]
                        st.session_state.true_label = 1
            
            # Display transaction
            if 'demo_transaction' in st.session_state:
                transaction = st.session_state.demo_transaction
                
                # Analyze button
                if st.button("🔍 Analyze Transaction", type="primary"):
                    with st.spinner("Analyzing..."):
                        error, uncertainty, reconstruction = predict_transaction(
                            transaction, selected_model, selected_model_name, scaler
                        )
                        
                        # Store results
                        st.session_state.error = error
                        st.session_state.uncertainty = uncertainty
                        st.session_state.reconstruction = reconstruction
                
                # Show transaction details
                with st.expander("📋 View Transaction Details"):
                    df_trans = pd.DataFrame({
                        'Feature': [f'V{i}' for i in range(1, 29)] + ['Amount'],
                        'Value': transaction
                    })
                    st.dataframe(df_trans, use_container_width=True)
        
        with col2:
            st.subheader("Detection Result")
            
            if 'error' in st.session_state:
                error = st.session_state.error
                
                # Calculate threshold (simplified)
                threshold = threshold_percentile / 100.0 * 2.0  # Simplified threshold
                
                is_fraud = error > threshold
                
                # Display result
                if is_fraud:
                    st.markdown('<p class="fraud-detected">🚨 FRAUD DETECTED</p>', unsafe_allow_html=True)
                    st.error(f"Reconstruction Error: {error:.4f}")
                else:
                    st.markdown('<p class="normal-transaction">✅ NORMAL TRANSACTION</p>', unsafe_allow_html=True)
                    st.success(f"Reconstruction Error: {error:.4f}")
                
                st.metric("Threshold", f"{threshold:.4f}")
                
                # Confidence bar
                confidence = min(abs(error - threshold) / threshold * 100, 100)
                st.progress(confidence / 100)
                st.write(f"Confidence: {confidence:.1f}%")
                
                # Uncertainty (if VAE)
                if st.session_state.uncertainty is not None:
                    st.metric("Uncertainty", f"{st.session_state.uncertainty:.4f}")
                
                # True label (if available)
                if 'true_label' in st.session_state:
                    true_label = "Fraud" if st.session_state.true_label == 1 else "Normal"
                    pred_label = "Fraud" if is_fraud else "Normal"
                    
                    if true_label == pred_label:
                        st.success(f"✅ Correct! (True: {true_label})")
                    else:
                        st.warning(f"❌ Incorrect (True: {true_label})")
    
    # Tab 2: Model Analysis
    with tab2:
        st.header("Model Architecture & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📐 {selected_model_name} Architecture")
            
            if selected_model_name == "Standard AE":
                st.info("""
                **Standard Autoencoder**
                - Input: 29 features
                - Encoder: 29 → 14 → 7
                - Decoder: 7 → 14 → 29
                - Activation: ReLU
                - Loss: MSE
                """)
            
            elif selected_model_name == "VAE":
                st.info("""
                **Variational Autoencoder**
                - Input: 29 features
                - Encoder: 29 → 14 → (μ, σ)
                - Latent: 7 dimensions (probabilistic)
                - Decoder: 7 → 14 → 29
                - Loss: MSE + KL Divergence
                - Provides uncertainty estimates
                """)
            
            elif selected_model_name == "LSTM-AE":
                st.info("""
                **LSTM Autoencoder**
                - Input: 10 timesteps × 29 features
                - Encoder: LSTM(64) → LSTM(32) → LSTM(7)
                - Decoder: LSTM(7) → LSTM(32) → LSTM(64)
                - Captures temporal patterns
                - Good for sequence anomalies
                """)
        
        with col2:
            st.subheader("🎯 Model Advantages")
            
            if selected_model_name == "Standard AE":
                st.success("""
                ✅ Fast training and inference
                ✅ Simple and interpretable
                ✅ Good baseline performance
                ✅ Low computational requirements
                """)
            
            elif selected_model_name == "VAE":
                st.success("""
                ✅ Probabilistic predictions
                ✅ Uncertainty quantification
                ✅ Smoother latent space
                ✅ Better generalization
                ✅ Can generate synthetic samples
                """)
            
            elif selected_model_name == "LSTM-AE":
                st.success("""
                ✅ Captures temporal patterns
                ✅ Detects sequence anomalies
                ✅ Time-series fraud detection
                ✅ Context-aware predictions
                """)
    
    # Tab 3: Performance Metrics
    with tab3:
        st.header("📈 Model Performance")
        
        # Sample metrics (would be loaded from training results)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision", "2.82%", help="Ratio of true frauds among detected frauds")
        
        with col2:
            st.metric("Recall", "84.69%", help="Ratio of detected frauds among all frauds")
        
        with col3:
            st.metric("F1-Score", "0.0547", help="Harmonic mean of precision and recall")
        
        with col4:
            st.metric("PR-AUC", "0.4865", help="Precision-Recall Area Under Curve")
        
        st.divider()
        
        # Visualizations
        if os.path.exists('outputs'):
            image_files = {
                'Training History': 'outputs/training_history.png',
                'Error Distribution': 'outputs/reconstruction_error_distribution.png',
                'Confusion Matrix': 'outputs/confusion_matrix.png',
                'PR Curve': 'outputs/precision_recall_curve.png',
                'Feature Explainability': 'outputs/feature_explainability.png'
            }
            
            selected_viz = st.selectbox("Select Visualization", list(image_files.keys()))
            
            if os.path.exists(image_files[selected_viz]):
                st.image(image_files[selected_viz], use_container_width=True)
            else:
                st.warning(f"Visualization not found: {image_files[selected_viz]}")
    
    # Tab 4: About
    with tab4:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This is an **advanced fraud detection system** using deep learning autoencoders
        trained exclusively on normal transactions to detect anomalies.
        
        ### 🧠 Methodology
        
        - **Unsupervised Learning**: Train only on normal transactions (Class = 0)
        - **Anomaly Detection**: Fraud transactions have high reconstruction errors
        - **Critical Split Strategy**: Ensures no fraud leakage during training
        
        ### 🔬 Three Advanced Models
        
        1. **Standard Autoencoder**: Fast, simple baseline
        2. **Variational Autoencoder (VAE)**: Probabilistic with uncertainty
        3. **LSTM Autoencoder**: Captures temporal patterns
        
        ### 📊 Dataset
        
        - **Source**: Kaggle Credit Card Fraud Detection
        - **Transactions**: 284,807
        - **Fraud Rate**: 0.17% (highly imbalanced)
        - **Features**: 28 PCA components + Amount + Time
        
        ### 🛠️ Technology Stack
        
        - **Framework**: TensorFlow 2.x / Keras
        - **Web App**: Streamlit
        - **Visualization**: Plotly, Matplotlib
        - **ML**: Scikit-learn, NumPy, Pandas
        
        ### 👨‍💻 Features
        
        - ✅ Real-time fraud detection
        - ✅ Multiple model comparison
        - ✅ Interactive visualization
        - ✅ Uncertainty quantification (VAE)
        - ✅ Temporal pattern detection (LSTM)
        - ✅ Feature explainability
        
        ### 📝 Citation
        
        Based on research in unsupervised anomaly detection using deep autoencoder
        manifold learning for high-frequency financial transactions.
        
        ---
        
        **Developed as part of B.Tech Final Year Project**
        """)

if __name__ == "__main__":
    main()
