#!/bin/bash
# Quick start script for advanced features

echo "🚀 Advanced Fraud Detection System - Setup & Launch"
echo "=" | head -c 80 && echo

# Install new dependencies
echo "📦 Installing new dependencies..."
pip install streamlit plotly

# Check if models exist
if [ ! -f "models/fraud_autoencoder.keras" ]; then
    echo "⚠️  No trained models found!"
    echo "Would you like to:"
    echo "  1. Train all advanced models (15-20 min)"
    echo "  2. Train standard model only (3-5 min)"
    echo "  3. Skip and launch web app anyway"
    read -p "Enter choice (1/2/3): " choice
    
    case $choice in
        1)
            echo "🏋️  Training all advanced models..."
            python3 train_advanced_models.py
            ;;
        2)
            echo "🏋️  Training standard model..."
            python3 train.py
            ;;
        3)
            echo "⏭️  Skipping training..."
            ;;
    esac
fi

# Launch web app
echo ""
echo "🌐 Launching Streamlit web application..."
echo "   The app will open at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py
