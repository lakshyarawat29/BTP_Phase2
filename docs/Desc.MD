Here is a curated project description and a step-by-step execution plan designed to meet the "A-Grade" criteria within your 3-day timeline.

### **Project Title**

**"The Truth Filter: Unsupervised Anomaly Detection in High-Frequency Financial Transactions via Deep Autoencoder Manifold Learning"**

### **Curated Project Description**

> *Use this for your project abstract or introduction.*

In the high-stakes domain of financial fraud detection, traditional supervised learning models fail two critical tests: they struggle with the severe class imbalance of real-world data (where fraud is \<0.2%), and they are blind to "zero-day" attack vectors they haven't been trained on. This project proposes a paradigm shift from *classification* to *unsupervised anomaly detection*. By training a Deep Autoencoder exclusively on legitimate transaction data, we force the model to learn the compressed latent manifold of "normality."

The resulting system acts as a "Truth Filter": legitimate transactions are reconstructed with high fidelity, while fraudulent patterns—violating the learned mathematical correlations—suffer high reconstruction errors. Unlike "black box" classifiers, this architecture offers inherent explainability; by analyzing the reconstruction error vector, we can isolate not just *that* a transaction is fraudulent, but *specifically which features* (e.g., unusual amount, abnormal location) triggered the alarm.

-----

### **Comprehensive To-Do List (3-Day Sprint)**

This roadmap is designed to get you from "Empty Folder" to "Final Report" efficiently.

#### **Phase 1: Setup & The "Critical Split" (Hours 1-4)**

*Goal: Prepare the data specifically for unsupervised learning. This is where most students fail by treating it like a standard classification problem.*

  - [ ] **Environment Setup:**
      - Install libraries: `pip install tensorflow pandas numpy scikit-learn matplotlib seaborn`
  - [ ] **Data Acquisition:**
      - Download the **Credit Card Fraud Detection** dataset from Kaggle (the `creditcard.csv` file).
  - [ ] **Preprocessing (Crucial Step):**
      - Load data into Pandas.
      - **Scale:** Apply `StandardScaler` or `RobustScaler` to the 'Amount' and 'Time' columns. (The other columns V1-V28 are already scaled).
      - **Drop:** Remove the 'Time' column if you want to simplify, or keep it if scaled.
  - [ ] **The "A-Grade" Split Strategy:**
      - Split data into `Training` (80%) and `Testing` (20%).
      - **Filter Training Data:** Create a new dataframe `X_train_normal` that contains **only** transactions where `Class == 0`. Drop the `Class` label column from this set.
      - **Prepare Test Data:** Keep `X_test` containing **both** Normal (0) and Fraud (1) rows. Keep the labels (`y_test`) separate for evaluation.

#### **Phase 2: Architecture & Training (Hours 5-10)**

*Goal: Build the "Bottleneck" that forces the model to learn patterns.*

  - [ ] **Build the Autoencoder (Keras/TensorFlow):**
      - Input Layer: Shape = `(29,)` (assuming 29 features).
      - Encoder Layers: Dense(14, activation='relu') $\rightarrow$ Dense(7, activation='relu').
      - **Bottleneck:** This 7-neuron layer is your "compressed knowledge."
      - Decoder Layers: Dense(14, activation='relu') $\rightarrow$ Dense(29, activation='linear' or 'sigmoid').
  - [ ] **Compile:**
      - Optimizer: `Adam` (learning rate=0.001).
      - Loss Function: `Mean Squared Error (MSE)` or `MAE`.
  - [ ] **Train:**
      - Fit the model on `X_train_normal` (Input = X, Target = X).
      - **Important:** Use `epochs=50` (or more) and `batch_size=32`.
      - Use `validation_data=(X_test, X_test)` to monitor overfitting.

#### **Phase 3: The "Money Shots" & Visualizations (Hours 11-16)**

*Goal: Generate the specific visuals that prove the concept works.*

  - [ ] **Generate Reconstruction Errors:**
      - Pass `X_test` through the trained model to get `reconstructions`.
      - Calculate the MSE Loss for each individual row: `np.mean(np.power(X_test - reconstructions, 2), axis=1)`.
  - [ ] **Visual 1: The "Separation" Histogram (Must Have):**
      - Plot a histogram of these error values.
      - Color code them: Normal (Blue) vs. Fraud (Red).
      - *Goal:* Show a massive blue peak near 0 and a red tail extending far to the right.
  - [ ] **Determine Threshold:**
      - Calculate the 95th percentile of the *training* error. Set this as your `threshold`.
      - Anything with `error > threshold` is predicted as Fraud.

#### **Phase 4: "A-Grade" Analytics (Hours 17-24)**

*Goal: Deep dive analysis that elevates the project beyond code.*

  - [ ] **Evaluation Metrics (Avoid Accuracy\!):**
      - Calculate **Precision**, **Recall**, and **F1-Score**.
      - Plot the **Precision-Recall Curve (AUPRC)**. *Note: A high Recall is more important than Precision for fraud (better to call you to check than to let money be stolen).*
  - [ ] **Visual 2: The Confusion Matrix:**
      - Plot the matrix using your chosen threshold.
  - [ ] **Feature-Wise Explainability (The "XAI" Component):**
      - Pick one specific fraudulent transaction (True Positive).
      - Calculate the absolute difference between Input and Output for *each feature* (V1, V2... Amount).
      - Plot a bar chart showing which feature had the highest error.
      - *Caption:* "The model flagged this transaction because Feature V14 deviated 400% from the expected normal behavior."

#### **Phase 5: Final Report Assembly (Last Day)**

*Goal: Synthesis.*

  - [ ] **Write the "Ethical Implications" Section:**
      - Discuss how training only on "normal" data might encode bias (e.g., if "normal" means "high spending," low spending might be flagged).
  - [ ] **Write the "Future Work" Section:**
      - Mention implementing **Variational Autoencoders (VAEs)** for a smoother latent space.
  - [ ] **Final Polish:**
      - Ensure all axes on graphs are labeled.
      - Add a "Management Summary" at the top for non-technical readers.

### **Quick Code Snippet for the "XAI" Step (Phase 4)**

This is the part most students miss. Use this to generate your "Why it's fraud" plot:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get the error vector for a single fraud instance (e.g., index 50)
fraud_idx = 50 
input_sample = X_test[fraud_idx]
reconstructed_sample = model.predict(input_sample.reshape(1, -1))
# Calculate absolute error per feature
feature_errors = np.abs(input_sample - reconstructed_sample)

# Plot
plt.bar(range(len(feature_errors)), feature_errors)
plt.title("Why was this flagged? (Reconstruction Error by Feature)")
plt.xlabel("Feature Index")
plt.ylabel("Error Magnitude")
plt.show()
```