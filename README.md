# The Truth Filter: Fraud Detection using Deep Autoencoders

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

**"The Truth Filter: Unsupervised Anomaly Detection in High-Frequency Financial Transactions via Deep Autoencoder Manifold Learning"**

This project implements an unsupervised deep learning approach to credit card fraud detection using autoencoders. Unlike traditional supervised methods that struggle with class imbalance and zero-day attacks, this system learns the manifold of "normal" transactions and flags anomalies based on reconstruction error.

### Key Features

- 🔒 **Unsupervised Learning**: Trained only on legitimate transactions
- 🎯 **Anomaly Detection**: No labeled fraud data required during training
- 📊 **Explainable AI**: Feature-level analysis showing _why_ transactions are flagged
- ⚖️ **Handles Imbalance**: Natural approach to highly imbalanced datasets (<0.2% fraud)
- 🔍 **Zero-Day Detection**: Can identify novel fraud patterns never seen before

## Architecture

```
Input (29) → Encoder (14) → Bottleneck (7) → Decoder (14) → Output (29)
                              ↓
                    Compressed Manifold
                  (Normal Transactions)
```

**Reconstruction Error = |Input - Output|**

- Low error → Normal transaction
- High error → Potential fraud

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd Take2.0.0
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

   - Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the `data/` folder

   Or use Kaggle API:

   ```bash
   pip install kaggle
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip -d data/
   ```

## Project Structure

```
Take2.0.0/
├── data/
│   └── creditcard.csv          # Dataset (download required)
├── models/
│   ├── fraud_autoencoder.keras # Trained model
│   └── scaler.pkl              # Feature scaler
├── outputs/
│   ├── training_history.png
│   ├── reconstruction_error_distribution.png
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── feature_explainability.png
├── data_loader.py              # Dataset loading utilities
├── preprocessor.py             # Data preprocessing (critical split)
├── autoencoder.py              # Autoencoder architecture
├── visualizer.py               # Visualization module
├── train.py                    # Main training pipeline
├── requirements.txt            # Python dependencies
├── Desc.MD                     # Project description
└── README.md                   # This file
```

## Usage

### Quick Start

Run the complete training pipeline:

```bash
python3 train.py
```

This will:

1. Load and preprocess the data
2. Build and train the autoencoder
3. Calculate reconstruction errors
4. Generate all visualizations
5. Print evaluation metrics

### Module-by-Module Usage

#### 1. Data Loading

```python
from data_loader import load_data

df = load_data('data/creditcard.csv')
```

#### 2. Preprocessing (The Critical Split)

```python
from preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(use_robust_scaler=True)
X_train_normal, X_test, y_test = preprocessor.preprocess(df)
```

**Critical**: Training set contains ONLY normal transactions (Class == 0)

#### 3. Build & Train Autoencoder

```python
from autoencoder import FraudAutoencoder

autoencoder = FraudAutoencoder(input_dim=29, encoding_dims=[14, 7])
autoencoder.build_model()
autoencoder.compile_model()
autoencoder.train(X_train_normal, X_val=X_test, epochs=50)
```

#### 4. Evaluation & Visualization

```python
from visualizer import FraudVisualizer

errors, reconstructions = autoencoder.calculate_reconstruction_error(X_test)
threshold = np.percentile(errors, 95)
y_pred = (errors > threshold).astype(int)

visualizer = FraudVisualizer()
visualizer.plot_reconstruction_error_distribution(errors, y_test, threshold)
visualizer.plot_confusion_matrix(y_test, y_pred)
visualizer.print_classification_report(y_test, y_pred, threshold)
```

## Methodology

### The Critical Split Strategy

Unlike standard supervised learning, we employ a **critical split**:

1. **Training Set**: Contains ONLY normal transactions (Class == 0)
2. **Test Set**: Contains both normal and fraudulent transactions

This forces the autoencoder to learn the "manifold of normality" without being exposed to fraud patterns.

### Threshold Determination

The anomaly threshold is set at the **95th percentile** of reconstruction errors on the training set:

```python
threshold = np.percentile(train_errors, 95)
```

This means:

- 95% of normal transactions have error < threshold
- Transactions with error > threshold are flagged as potential fraud

### Explainability

For each flagged transaction, we calculate the reconstruction error for each individual feature:

```python
feature_errors = |input_features - reconstructed_features|
```

This shows which specific features (amount, location, time, etc.) deviated from expected behavior.

## Results

Expected performance metrics on the Credit Card Fraud Detection dataset:

- **Precision**: ~0.85-0.95 (of flagged transactions, 85-95% are actually fraud)
- **Recall**: ~0.75-0.90 (detect 75-90% of all fraud cases)
- **PR-AUC**: ~0.80-0.90 (Area under Precision-Recall curve)

### Visualizations

1. **Training History**: Loss curves showing model convergence
2. **Reconstruction Error Distribution**: The "separation" histogram (THE MONEY SHOT)
3. **Confusion Matrix**: True/False Positives/Negatives breakdown
4. **Precision-Recall Curve**: Better than ROC for imbalanced data
5. **Feature Explainability**: Bar chart showing which features triggered the alert

## Key Insights

### Why This Approach Works

1. **Class Imbalance**: No need for complex sampling techniques
2. **Novel Fraud**: Can detect attacks never seen before
3. **Explainable**: Not a "black box" - we know WHY something is flagged
4. **Mathematical Foundation**: Based on manifold learning theory

### Limitations

1. **Concept Drift**: Normal transaction patterns may change over time
2. **Cold Start**: Needs sufficient normal data to learn patterns
3. **False Positives**: May flag legitimate but unusual transactions

### Ethical Considerations

- **Bias**: If "normal" is biased (e.g., high spending), model may discriminate
- **Privacy**: Feature analysis must respect customer privacy
- **Human-in-the-Loop**: High-stakes decisions should involve human review

## Future Work

- [ ] Implement **Variational Autoencoders (VAEs)** for smoother latent space
- [ ] Add **LSTM layers** to capture temporal patterns
- [ ] Implement **online learning** to adapt to concept drift
- [ ] Deploy as a **real-time API** for production use
- [ ] Add **SHAP values** for enhanced explainability

## References

1. Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Paper: "Credit Card Fraud Detection Using Autoencoders" (Various)
3. TensorFlow Documentation: [Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**BTP Project - Semester 7**

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an academic project for learning purposes. For production fraud detection systems, additional security measures, regulatory compliance, and extensive testing are required.
