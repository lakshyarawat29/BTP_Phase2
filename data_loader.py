"""
Data Loader Module for Credit Card Fraud Detection
Downloads and loads the Kaggle Credit Card Fraud Detection dataset
"""

import pandas as pd
import os

def download_instructions():
    """
    Prints instructions for downloading the dataset from Kaggle
    """
    print("=" * 80)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\nPlease follow these steps to download the dataset:")
    print("\n1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Click 'Download' button (you may need to sign in to Kaggle)")
    print("3. Extract 'creditcard.csv' from the downloaded zip file")
    print("4. Place 'creditcard.csv' in the 'data/' folder of this project")
    print("\nAlternatively, you can use Kaggle API:")
    print("  $ pip install kaggle")
    print("  $ kaggle datasets download -d mlg-ulb/creditcardfraud")
    print("  $ unzip creditcardfraud.zip -d data/")
    print("\n" + "=" * 80)

def load_data(filepath='data/creditcard.csv'):
    """
    Loads the credit card dataset from CSV file
    
    Args:
        filepath (str): Path to the creditcard.csv file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    if not os.path.exists(filepath):
        print(f"\n❌ Dataset not found at: {filepath}\n")
        download_instructions()
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Display basic info
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"\nFraud percentage: {(df['Class'].sum() / len(df) * 100):.4f}%")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Try to load the data
    try:
        data = load_data()
        print("\n✨ Data is ready for preprocessing!")
    except FileNotFoundError:
        print("\n⚠️  Please download the dataset first and run again.")
