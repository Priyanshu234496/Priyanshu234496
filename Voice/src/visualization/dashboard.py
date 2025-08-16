"""
Streamlit Dashboard for Multi-Modal Earnings Call Forecaster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import logging

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.text_analysis.analyzer import TextAnalyzer
from src.audio_analysis.analyzer import AudioAnalyzer
from src.modeling.predictor import EarningsCallPredictor
from config.config import DASHBOARD_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_SETTINGS['page_title'],
    page_icon=DASHBOARD_SETTINGS['page_icon'],
    layout=DASHBOARD_SETTINGS['layout'],
    initial_sidebar_state=DASHBOARD_SETTINGS['initial_sidebar_state']
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EarningsCallDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.text_analyzer = TextAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.predictor = EarningsCallPredictor()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None
    
    def run(self):
        """Run the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🎤 Multi-Modal Earnings Call Forecaster</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🎯 Prediction", "📈 Insights", "⚙️ Settings"])
        
        with tab1:
            self.analysis_tab()
        
        with tab2:
            self.prediction_tab()
        
        with tab3:
            self.insights_tab()
        
        with tab4:
            self.settings_tab()
    
    def create_sidebar(self):
        """Create the sidebar with navigation and file upload"""
        st.sidebar.title("📁 Data Input")
        
        # File upload section
        st.sidebar.subheader("Upload Files")
        
        # Transcript upload
        transcript_file = st.sidebar.file_uploader(
            "Upload Transcript (TXT)",
            type=['txt'],
            help="Upload earnings call transcript text file"
        )
        
        # Audio upload
        audio_file = st.sidebar.file_uploader(
            "Upload Audio (MP3/WAV)",
            type=['mp3', 'wav', 'm4a'],
            help="Upload earnings call audio recording"
        )
        
        # Company information
        st.sidebar.subheader("Company Information")
        company_symbol = st.sidebar.text_input(
            "Company Symbol",
            value="AAPL",
            help="Stock symbol (e.g., AAPL, MSFT)"
        )
        
        call_date = st.sidebar.date_input(
            "Call Date",
            value=datetime.now(),
            help="Date of the earnings call"
        )
        
        # Analysis options
        st.sidebar.subheader("Analysis Options")
        include_audio = st.sidebar.checkbox(
            "Include Audio Analysis",
            value=True,
            help="Perform audio feature extraction"
        )
        
        include_speakers = st.sidebar.checkbox(
            "Speaker Analysis",
            value=True,
            help="Analyze individual speakers"
        )
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            if transcript_file is not None:
                self.run_analysis(transcript_file, audio_file, company_symbol, call_date, include_audio, include_speakers)
            else:
                st.error("Please upload a transcript file to begin analysis.")
        
        # Sample data
        st.sidebar.subheader("Sample Data")
        if st.sidebar.button("📋 Load Sample Data"):
            self.load_sample_data()
    
    def run_analysis(self, transcript_file, audio_file, company_symbol, call_date, include_audio, include_speakers):
        """Run the complete analysis pipeline"""
        try:
            with st.spinner("🔍 Analyzing transcript..."):
                # Read transcript
                transcript_text = transcript_file.read().decode('utf-8')
                
                # Text analysis
                text_analysis = self.text_analyzer.analyze_transcript(transcript_text)
                
                # Audio analysis
                audio_analysis = None
                if include_audio and audio_file is not None:
                    with st.spinner("🎵 Analyzing audio..."):
                        # Save audio file temporarily
                        temp_audio_path = Path("temp_audio.wav")
                        with open(temp_audio_path, "wb") as f:
                            f.write(audio_file.read())
                        
                        audio_analysis = self.audio_analyzer.analyze_audio(temp_audio_path)
                        
                        # Clean up
                        temp_audio_path.unlink(missing_ok=True)
                
                # Store results
                st.session_state.analysis_results = {
                    'text_analysis': text_analysis,
                    'audio_analysis': audio_analysis,
                    'company_symbol': company_symbol,
                    'call_date': call_date,
                    'transcript_text': transcript_text
                }
                
                st.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {e}")
    
    def analysis_tab(self):
        """Display analysis results"""
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please upload files and run analysis from the sidebar.")
            return
        
        results = st.session_state.analysis_results
        
        # Company and date info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", results['company_symbol'])
        with col2:
            st.metric("Call Date", results['call_date'].strftime('%Y-%m-%d'))
        with col3:
            st.metric("Analysis Type", "Multi-Modal" if results['audio_analysis'] else "Text-Only")
        
        # Text Analysis Results
        st.subheader("📝 Text Analysis")
        text_features = results['text_analysis']['features']
        text_summary = results['text_analysis']['summary']
        
        # Text metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", f"{text_features.get('word_count', 0):,}")
        with col2:
            st.metric("Sentiment", text_summary.get('overall_sentiment', 'Unknown'))
        with col3:
            st.metric("Readability", text_summary.get('readability_level', 'Unknown'))
        with col4:
            st.metric("Uncertainty", text_summary.get('uncertainty_level', 'Unknown'))
        
        # Text insights
        st.subheader("💡 Text Insights")
        for insight in results['text_analysis']['insights']:
            st.markdown(f'<div class="insight-box">💭 {insight}</div>', unsafe_allow_html=True)
        
        # Text feature charts
        self.create_text_charts(text_features)
        
        # Audio Analysis Results
        if results['audio_analysis']:
            st.subheader("🎵 Audio Analysis")
            audio_features = results['audio_analysis']['features']
            audio_summary = results['audio_analysis']['summary']
            
            # Audio metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{audio_summary.get('duration_minutes', 0):.1f} min")
            with col2:
                st.metric("Speech Rate", f"{audio_summary.get('speech_rate', 0):.0f} WPM")
            with col3:
                st.metric("Voice Quality", audio_summary.get('voice_quality', 'Unknown'))
            with col4:
                st.metric("Stress Level", audio_summary.get('stress_level', 'Unknown'))
            
            # Audio insights
            st.subheader("🎧 Audio Insights")
            for insight in results['audio_analysis']['insights']:
                st.markdown(f'<div class="insight-box">🎵 {insight}</div>', unsafe_allow_html=True)
            
            # Audio feature charts
            self.create_audio_charts(audio_features)
    
    def prediction_tab(self):
        """Display prediction results"""
        st.header("🎯 Volatility Prediction")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please run analysis first to generate predictions.")
            return
        
        # Prediction controls
        col1, col2 = st.columns(2)
        with col1:
            prediction_horizon = st.selectbox(
                "Prediction Horizon",
                ["1 Day", "1 Week", "2 Weeks", "1 Month"],
                help="Time horizon for volatility prediction"
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence interval for prediction"
            )
        
        # Run prediction
        if st.button("🔮 Generate Prediction", type="primary"):
            with st.spinner("🤖 Generating prediction..."):
                try:
                    prediction = self.predictor.predict(
                        st.session_state.analysis_results,
                        horizon=prediction_horizon,
                        confidence=confidence_level
                    )
                    
                    st.session_state.prediction_results = prediction
                    st.success("✅ Prediction generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating prediction: {str(e)}")
        
        # Display prediction results
        if st.session_state.prediction_results:
            self.display_prediction_results(st.session_state.prediction_results)
    
    def insights_tab(self):
        """Display insights and trends"""
        st.header("📈 Insights & Trends")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please run analysis first to view insights.")
            return
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        self.create_feature_importance_chart()
        
        # Comparative analysis
        st.subheader("📊 Comparative Analysis")
        self.create_comparative_charts()
        
        # Recommendations
        st.subheader("💡 Recommendations")
        self.display_recommendations()
    
    def settings_tab(self):
        """Display settings and configuration"""
        st.header("⚙️ Settings & Configuration")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "XGBoost Estimators",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Number of boosting rounds"
            )
            
            st.number_input(
                "Max Depth",
                min_value=3,
                max_value=10,
                value=6,
                help="Maximum tree depth"
            )
        
        with col2:
            st.number_input(
                "Learning Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                help="Learning rate for boosting"
            )
            
            st.number_input(
                "CV Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Cross-validation folds"
            )
        
        # Feature settings
        st.subheader("🔧 Feature Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Include Audio Features", value=True)
            st.checkbox("Include Speaker Analysis", value=True)
            st.checkbox("Include Topic Modeling", value=True)
        
        with col2:
            st.checkbox("Include Formant Analysis", value=True)
            st.checkbox("Include Rhythm Analysis", value=True)
            st.checkbox("Include Stress Indicators", value=True)
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
    
    def create_text_charts(self, text_features):
        """Create text analysis charts"""
        # Sentiment distribution
        sentiment_data = {
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Score': [
                text_features.get('positive_sentiment', 0),
                text_features.get('negative_sentiment', 0),
                text_features.get('neutral_sentiment', 0)
            ]
        }
        
        fig_sentiment = px.pie(
            sentiment_data, 
            values='Score', 
            names='Sentiment',
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#808080'
            }
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Financial lexicon features
        lexicon_features = {
            'Category': ['Uncertainty', 'Litigious', 'Negative', 'Positive'],
            'Count': [
                text_features.get('uncertainty_count', 0),
                text_features.get('litigious_count', 0),
                text_features.get('negative_count', 0),
                text_features.get('positive_count', 0)
            ]
        }
        
        fig_lexicon = px.bar(
            lexicon_features,
            x='Category',
            y='Count',
            title="Financial Lexicon Analysis",
            color='Category'
        )
        st.plotly_chart(fig_lexicon, use_container_width=True)
    
    def create_audio_charts(self, audio_features):
        """Create audio analysis charts"""
        # Pitch analysis
        pitch_data = {
            'Metric': ['Mean', 'Std', 'Min', 'Max'],
            'Value': [
                audio_features.get('pitch_mean', 0),
                audio_features.get('pitch_std', 0),
                audio_features.get('pitch_min', 0),
                audio_features.get('pitch_max', 0)
            ]
        }
        
        fig_pitch = px.bar(
            pitch_data,
            x='Metric',
            y='Value',
            title="Pitch Analysis",
            color='Metric'
        )
        st.plotly_chart(fig_pitch, use_container_width=True)
        
        # Voice quality indicators
        quality_data = {
            'Indicator': ['Jitter', 'Shimmer', 'HNR'],
            'Value': [
                audio_features.get('jitter_local', 0),
                audio_features.get('shimmer_local', 0),
                audio_features.get('hnr_mean', 0)
            ]
        }
        
        fig_quality = px.bar(
            quality_data,
            x='Indicator',
            y='Value',
            title="Voice Quality Indicators",
            color='Indicator'
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    def display_prediction_results(self, prediction):
        """Display prediction results"""
        # Prediction metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Predicted Volatility",
                f"{prediction.get('volatility', 0):.2%}",
                delta=f"{prediction.get('volatility_change', 0):.2%}"
            )
        with col2:
            st.metric(
                "Confidence",
                f"{prediction.get('confidence', 0):.1%}"
            )
        with col3:
            st.metric(
                "Risk Level",
                prediction.get('risk_level', 'Unknown')
            )
        
        # Prediction chart
        if 'prediction_timeline' in prediction:
            fig_prediction = px.line(
                prediction['prediction_timeline'],
                x='Date',
                y='Volatility',
                title="Volatility Prediction Timeline"
            )
            st.plotly_chart(fig_prediction, use_container_width=True)
        
        # Feature importance for prediction
        if 'feature_importance' in prediction:
            st.subheader("🎯 Prediction Feature Importance")
            importance_df = pd.DataFrame(prediction['feature_importance'])
            fig_importance = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title="Feature Importance for Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        # Sample feature importance data
        features = [
            'Management Sentiment', 'Audio Stress Score', 'Forward-Looking Statements',
            'Voice Quality', 'Uncertainty Ratio', 'Speech Rate', 'Pitch Variability'
        ]
        importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.08, 0.02]
        
        fig = px.bar(
            x=features,
            y=importance,
            title="Overall Feature Importance",
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_comparative_charts(self):
        """Create comparative analysis charts"""
        # Sample comparative data
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        avg_volatility = [0.15, 0.18, 0.12, 0.22, 0.16]
        avg_sentiment = [0.65, 0.58, 0.72, 0.45, 0.61]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Volatility by Company', 'Average Sentiment by Company')
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=avg_volatility, name='Volatility'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=companies, y=avg_sentiment, name='Sentiment'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_recommendations(self):
        """Display recommendations based on analysis"""
        recommendations = [
            "📈 Consider increasing forward-looking statements for better investor confidence",
            "🎵 Monitor vocal stress indicators during future calls",
            "📊 Focus on reducing uncertainty language in prepared remarks",
            "🎯 Implement more structured Q&A sessions",
            "📝 Maintain consistent speech rate throughout the call"
        ]
        
        for rec in recommendations:
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_transcript = """
        Operator: Good afternoon and welcome to Apple's Q4 2023 Earnings Call. 
        
        Tim Cook (CEO): Thank you. Good afternoon everyone. We're very pleased to report 
        another strong quarter with record revenue of $89.5 billion. Our iPhone business 
        continues to perform exceptionally well, and we're seeing strong demand across 
        all our product categories.
        
        Luca Maestri (CFO): Thank you Tim. Let me provide some additional financial details. 
        Our gross margin was 44.3%, which was above our guidance range. We generated 
        $24.1 billion in operating cash flow and returned $25.1 billion to shareholders 
        through dividends and share repurchases.
        
        Analyst Question: Tim, can you comment on the supply chain situation and what 
        you're seeing for the holiday quarter?
        
        Tim Cook: We're seeing some improvement in supply chain constraints, but we 
        expect some challenges to continue into the holiday quarter. We're working 
        closely with our suppliers to minimize any impact on our customers.
        """
        
        st.session_state.sample_transcript = sample_transcript
        st.success("✅ Sample data loaded! You can now run analysis with the sample transcript.")

def main():
    """Main function to run the dashboard"""
    dashboard = EarningsCallDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
