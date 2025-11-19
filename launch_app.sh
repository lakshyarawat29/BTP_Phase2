#!/bin/bash
# Quick launch script for web app

echo "🚀 Launching Fraud Detection Web Application"
echo ""

# Check if model exists
if [ ! -f "models/fraud_autoencoder.keras" ]; then
    echo "❌ No trained model found!"
    echo "Please train a model first: python3 train.py"
    exit 1
fi

echo "✅ Found trained model: fraud_autoencoder.keras"
echo ""
echo "🌐 Starting Streamlit server..."
echo "   The app will open at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""
echo "⚠️  When prompted for email, just press Enter to skip"
echo ""

# Launch streamlit
streamlit run app.py --server.headless=true 2>/dev/null || streamlit run app.py
