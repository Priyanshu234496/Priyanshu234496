# 🚀 Quick Start Guide

Get up and running with the Multi-Modal Earnings Call Forecaster in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- Chrome browser (for web scraping)
- Git

## ⚡ Quick Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Voice

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Models

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Run the Dashboard

```bash
# Option 1: Use the run script
python run_dashboard.py

# Option 2: Run directly with Streamlit
streamlit run src/visualization/dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 🎯 First Steps

### 1. Load Sample Data
- Click "📋 Load Sample Data" in the sidebar
- This loads a sample Apple earnings call transcript

### 2. Run Analysis
- Click "🚀 Run Analysis" to analyze the sample data
- View results in the "📊 Analysis" tab

### 3. Generate Predictions
- Go to the "🎯 Prediction" tab
- Click "🔮 Generate Prediction" to see volatility forecasts

## 📁 Project Structure

```
Voice/
├── src/                    # Source code
│   ├── data_acquisition/   # Web scraping
│   ├── text_analysis/      # NLP analysis
│   ├── audio_analysis/     # Audio processing
│   ├── modeling/           # ML models
│   └── visualization/      # Dashboard
├── data/                   # Data storage
├── config/                 # Configuration
├── requirements.txt        # Dependencies
└── run_dashboard.py        # Quick start script
```

## 🔧 Configuration

Edit `config/config.py` to customize:
- Target companies
- Audio processing settings
- Model parameters
- Feature extraction options

## 📊 Key Features

### Text Analysis
- ✅ Financial lexicon analysis (Loughran-McDonald)
- ✅ Sentiment analysis
- ✅ Forward-looking statement detection
- ✅ Readability metrics
- ✅ Speaker analysis

### Audio Analysis
- ✅ Pitch analysis (fundamental frequency)
- ✅ Jitter and shimmer measurement
- ✅ Speech rate analysis
- ✅ Voice quality assessment
- ✅ Stress indicators

### Prediction
- ✅ Volatility forecasting
- ✅ Feature importance analysis
- ✅ Confidence intervals
- ✅ Risk level assessment

## 🎨 Dashboard Features

- **📊 Analysis Tab**: View detailed text and audio analysis results
- **🎯 Prediction Tab**: Generate and visualize volatility predictions
- **📈 Insights Tab**: Explore feature importance and trends
- **⚙️ Settings Tab**: Configure model parameters

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Audio Processing Errors**
   - Ensure you have audio files in supported formats (MP3, WAV, M4A)
   - Check that audio files are not corrupted

4. **Dashboard Won't Start**
   ```bash
   # Check if port 8501 is available
   # Or specify a different port:
   streamlit run src/visualization/dashboard.py --server.port 8502
   ```

### Getting Help

- Check the logs in the terminal for error messages
- Ensure all dependencies are installed correctly
- Verify that audio files are in supported formats

## 🔄 Next Steps

1. **Add Your Own Data**
   - Upload earnings call transcripts (TXT format)
   - Upload audio recordings (MP3/WAV/M4A format)

2. **Customize Analysis**
   - Modify feature extraction in `config/config.py`
   - Add new companies to the target list

3. **Train Custom Models**
   - Collect more earnings call data
   - Implement the training pipeline in `src/modeling/trainer.py`

4. **Extend Functionality**
   - Add new audio features
   - Implement additional text analysis techniques
   - Create custom visualizations

## 📚 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Explore the source code in the `src/` directory
- Check configuration options in `config/config.py`

---

**Happy Forecasting! 🎤📈**
