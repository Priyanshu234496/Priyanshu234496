"""
Main text analyzer for earnings call transcripts
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

from config.config import LOUGHRAN_MCDONALD_LEXICONS, FEATURE_SETTINGS

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """
    Comprehensive text analyzer for earnings call transcripts
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the text analyzer
        
        Args:
            spacy_model: spaCy model to use for NLP processing
        """
        self.spacy_model = spacy_model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found. Please install it.")
            self.nlp = None
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Compile regex patterns for forward-looking statements
        self.fls_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in FEATURE_SETTINGS['text_features']['forward_looking_patterns']
        ]
    
    def extract_features(self, transcript: str, speakers: Optional[List[Dict]] = None) -> Dict:
        """
        Extract comprehensive text features from earnings call transcript
        
        Args:
            transcript: Full transcript text
            speakers: List of speaker segments (optional)
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        try:
            # Basic text statistics
            features.update(self._extract_basic_stats(transcript))
            
            # Readability metrics
            features.update(self._extract_readability_metrics(transcript))
            
            # Financial lexicon analysis
            features.update(self._extract_financial_lexicon_features(transcript))
            
            # Forward-looking statement analysis
            features.update(self._extract_forward_looking_features(transcript))
            
            # Sentiment analysis
            features.update(self._extract_sentiment_features(transcript))
            
            # Speaker-specific features (if speakers provided)
            if speakers:
                features.update(self._extract_speaker_features(speakers))
            
            # Linguistic complexity features
            if self.nlp:
                features.update(self._extract_linguistic_complexity(transcript))
            
            # Uncertainty and hedging features
            features.update(self._extract_uncertainty_features(transcript))
            
            # Question-answer analysis
            features.update(self._extract_qa_features(transcript))
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            features = self._get_default_features()
        
        return features
    
    def _extract_basic_stats(self, transcript: str) -> Dict:
        """Extract basic text statistics"""
        words = transcript.split()
        sentences = transcript.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / max(len(words), 1),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
    
    def _extract_readability_metrics(self, transcript: str) -> Dict:
        """Extract readability metrics"""
        try:
            return {
                'gunning_fog': textstat.gunning_fog(transcript),
                'flesch_reading_ease': textstat.flesch_reading_ease(transcript),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(transcript),
                'smog_index': textstat.smog_index(transcript),
                'automated_readability_index': textstat.automated_readability_index(transcript),
                'coleman_liau_index': textstat.coleman_liau_index(transcript),
                'linsear_write_formula': textstat.linsear_write_formula(transcript),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(transcript)
            }
        except Exception as e:
            logger.warning(f"Error calculating readability metrics: {e}")
            return {metric: 0 for metric in [
                'gunning_fog', 'flesch_reading_ease', 'flesch_kincaid_grade',
                'smog_index', 'automated_readability_index', 'coleman_liau_index',
                'linsear_write_formula', 'dale_chall_readability_score'
            ]}
    
    def _extract_financial_lexicon_features(self, transcript: str) -> Dict:
        """Extract features based on Loughran-McDonald financial lexicons"""
        transcript_lower = transcript.lower()
        words = transcript_lower.split()
        word_count = len(words)
        
        features = {}
        
        for category, lexicon_words in LOUGHRAN_MCDONALD_LEXICONS.items():
            # Count occurrences
            count = sum(1 for word in words if word in lexicon_words)
            
            # Calculate ratios
            features[f'{category}_count'] = count
            features[f'{category}_ratio'] = count / max(word_count, 1)
            
            # Calculate normalized frequency (per 1000 words)
            features[f'{category}_per_1000'] = (count / max(word_count, 1)) * 1000
        
        return features
    
    def _extract_forward_looking_features(self, transcript: str) -> Dict:
        """Extract forward-looking statement features"""
        features = {}
        
        # Count forward-looking statements
        fls_count = 0
        fls_sentences = []
        
        sentences = transcript.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for forward-looking patterns
            for pattern in self.fls_patterns:
                if pattern.search(sentence):
                    fls_count += 1
                    fls_sentences.append(sentence)
                    break
        
        features['forward_looking_count'] = fls_count
        features['forward_looking_ratio'] = fls_count / max(len(sentences), 1)
        
        # Analyze sentiment of forward-looking statements
        if fls_sentences:
            fls_text = '. '.join(fls_sentences)
            fls_sentiment = self.sentiment_analyzer.polarity_scores(fls_text)
            features['fls_positive_sentiment'] = fls_sentiment['pos']
            features['fls_negative_sentiment'] = fls_sentiment['neg']
            features['fls_neutral_sentiment'] = fls_sentiment['neu']
            features['fls_compound_sentiment'] = fls_sentiment['compound']
        else:
            features.update({
                'fls_positive_sentiment': 0,
                'fls_negative_sentiment': 0,
                'fls_neutral_sentiment': 0,
                'fls_compound_sentiment': 0
            })
        
        return features
    
    def _extract_sentiment_features(self, transcript: str) -> Dict:
        """Extract sentiment analysis features"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(transcript)
        
        return {
            'positive_sentiment': sentiment_scores['pos'],
            'negative_sentiment': sentiment_scores['neg'],
            'neutral_sentiment': sentiment_scores['neu'],
            'compound_sentiment': sentiment_scores['compound'],
            'sentiment_polarity': abs(sentiment_scores['compound'])
        }
    
    def _extract_speaker_features(self, speakers: List[Dict]) -> Dict:
        """Extract speaker-specific features"""
        features = {}
        
        if not speakers:
            return features
        
        # Separate management and analyst speakers
        management_speakers = []
        analyst_speakers = []
        
        for speaker in speakers:
            name = speaker.get('name', '').lower()
            text = speaker.get('text', '')
            
            # Simple heuristic to identify management vs analysts
            if any(title in name for title in ['ceo', 'cfo', 'cto', 'president', 'chairman', 'executive']):
                management_speakers.append(text)
            elif any(title in name for title in ['analyst', 'question', 'q&a']):
                analyst_speakers.append(text)
            else:
                # Default to management if unclear
                management_speakers.append(text)
        
        # Management features
        if management_speakers:
            management_text = ' '.join(management_speakers)
            mgmt_sentiment = self.sentiment_analyzer.polarity_scores(management_text)
            mgmt_lexicon = self._extract_financial_lexicon_features(management_text)
            
            features.update({
                'management_positive_sentiment': mgmt_sentiment['pos'],
                'management_negative_sentiment': mgmt_sentiment['neg'],
                'management_compound_sentiment': mgmt_sentiment['compound'],
                'management_uncertainty_ratio': mgmt_lexicon.get('uncertainty_ratio', 0),
                'management_forward_looking_ratio': self._extract_forward_looking_features(management_text).get('forward_looking_ratio', 0)
            })
        
        # Analyst features
        if analyst_speakers:
            analyst_text = ' '.join(analyst_speakers)
            analyst_sentiment = self.sentiment_analyzer.polarity_scores(analyst_text)
            
            features.update({
                'analyst_positive_sentiment': analyst_sentiment['pos'],
                'analyst_negative_sentiment': analyst_sentiment['neg'],
                'analyst_compound_sentiment': analyst_sentiment['compound']
            })
        
        # Speaker interaction features
        features['speaker_count'] = len(speakers)
        features['management_analyst_ratio'] = len(management_speakers) / max(len(analyst_speakers), 1)
        
        return features
    
    def _extract_linguistic_complexity(self, transcript: str) -> Dict:
        """Extract linguistic complexity features using spaCy"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(transcript)
        
        # Part-of-speech ratios
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        features = {
            'noun_ratio': pos_counts.get('NOUN', 0) / max(total_tokens, 1),
            'verb_ratio': pos_counts.get('VERB', 0) / max(total_tokens, 1),
            'adj_ratio': pos_counts.get('ADJ', 0) / max(total_tokens, 1),
            'adv_ratio': pos_counts.get('ADV', 0) / max(total_tokens, 1),
            'pronoun_ratio': pos_counts.get('PRON', 0) / max(total_tokens, 1)
        }
        
        # Named entity features
        entity_types = Counter([ent.label_ for ent in doc.ents])
        features['named_entity_count'] = len(doc.ents)
        features['org_entity_ratio'] = entity_types.get('ORG', 0) / max(len(doc.ents), 1)
        features['money_entity_ratio'] = entity_types.get('MONEY', 0) / max(len(doc.ents), 1)
        features['date_entity_ratio'] = entity_types.get('DATE', 0) / max(len(doc.ents), 1)
        
        # Dependency complexity
        features['avg_dependency_distance'] = np.mean([
            abs(token.i - token.head.i) for token in doc
        ]) if doc else 0
        
        return features
    
    def _extract_uncertainty_features(self, transcript: str) -> Dict:
        """Extract uncertainty and hedging features"""
        uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'potentially', 'might', 'could',
            'would', 'should', 'may', 'can', 'uncertain', 'unclear', 'unknown',
            'depends', 'depends on', 'subject to', 'contingent', 'conditional'
        ]
        
        hedging_words = [
            'sort of', 'kind of', 'somewhat', 'relatively', 'fairly',
            'quite', 'rather', 'approximately', 'roughly', 'about'
        ]
        
        transcript_lower = transcript.lower()
        words = transcript_lower.split()
        word_count = len(words)
        
        # Count uncertainty and hedging words
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        hedging_count = sum(1 for word in words if word in hedging_words)
        
        # Count hedging phrases
        hedging_phrases = sum(1 for phrase in hedging_words if phrase in transcript_lower)
        
        return {
            'uncertainty_word_count': uncertainty_count,
            'uncertainty_word_ratio': uncertainty_count / max(word_count, 1),
            'hedging_word_count': hedging_count + hedging_phrases,
            'hedging_word_ratio': (hedging_count + hedging_phrases) / max(word_count, 1),
            'total_uncertainty_ratio': (uncertainty_count + hedging_count + hedging_phrases) / max(word_count, 1)
        }
    
    def _extract_qa_features(self, transcript: str) -> Dict:
        """Extract question-answer interaction features"""
        # Count questions
        question_patterns = [
            r'\?',  # Question marks
            r'\b(what|when|where|who|why|how)\b',  # Question words
            r'\b(can you|could you|would you|will you)\b',  # Question phrases
        ]
        
        question_count = 0
        for pattern in question_patterns:
            question_count += len(re.findall(pattern, transcript, re.IGNORECASE))
        
        # Count analyst questions vs management responses
        lines = transcript.split('\n')
        analyst_questions = 0
        management_responses = 0
        
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['analyst', 'question', 'q&a']):
                analyst_questions += 1
            elif any(word in line_lower for word in ['ceo', 'cfo', 'management', 'response']):
                management_responses += 1
        
        return {
            'question_count': question_count,
            'question_density': question_count / max(len(transcript.split()), 1),
            'analyst_questions': analyst_questions,
            'management_responses': management_responses,
            'qa_ratio': analyst_questions / max(management_responses, 1)
        }
    
    def _get_default_features(self) -> Dict:
        """Return default feature values when extraction fails"""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'unique_words': 0,
            'lexical_diversity': 0,
            'avg_word_length': 0,
            'gunning_fog': 0,
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'positive_sentiment': 0,
            'negative_sentiment': 0,
            'neutral_sentiment': 0,
            'compound_sentiment': 0,
            'forward_looking_count': 0,
            'forward_looking_ratio': 0,
            'uncertainty_ratio': 0,
            'question_count': 0
        }
    
    def analyze_transcript(self, transcript: str, speakers: Optional[List[Dict]] = None) -> Dict:
        """
        Comprehensive transcript analysis with detailed breakdown
        
        Args:
            transcript: Full transcript text
            speakers: List of speaker segments (optional)
            
        Returns:
            Dictionary with detailed analysis results
        """
        analysis = {
            'features': self.extract_features(transcript, speakers),
            'summary': {},
            'insights': []
        }
        
        # Generate summary statistics
        features = analysis['features']
        
        analysis['summary'] = {
            'total_words': features.get('word_count', 0),
            'readability_level': self._get_readability_level(features.get('flesch_reading_ease', 0)),
            'overall_sentiment': self._get_sentiment_label(features.get('compound_sentiment', 0)),
            'uncertainty_level': self._get_uncertainty_level(features.get('uncertainty_ratio', 0)),
            'forward_looking_intensity': self._get_fls_intensity(features.get('forward_looking_ratio', 0))
        }
        
        # Generate insights
        analysis['insights'] = self._generate_insights(features)
        
        return analysis
    
    def _get_readability_level(self, flesch_score: float) -> str:
        """Convert Flesch reading ease score to level"""
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound sentiment score to label"""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def _get_uncertainty_level(self, uncertainty_ratio: float) -> str:
        """Convert uncertainty ratio to level"""
        if uncertainty_ratio >= 0.05:
            return "High"
        elif uncertainty_ratio >= 0.02:
            return "Medium"
        else:
            return "Low"
    
    def _get_fls_intensity(self, fls_ratio: float) -> str:
        """Convert forward-looking statement ratio to intensity"""
        if fls_ratio >= 0.1:
            return "High"
        elif fls_ratio >= 0.05:
            return "Medium"
        else:
            return "Low"
    
    def _generate_insights(self, features: Dict) -> List[str]:
        """Generate insights from extracted features"""
        insights = []
        
        # Sentiment insights
        compound_sentiment = features.get('compound_sentiment', 0)
        if compound_sentiment > 0.3:
            insights.append("Strong positive sentiment detected in the call")
        elif compound_sentiment < -0.3:
            insights.append("Strong negative sentiment detected in the call")
        
        # Uncertainty insights
        uncertainty_ratio = features.get('uncertainty_ratio', 0)
        if uncertainty_ratio > 0.05:
            insights.append("High level of uncertainty language detected")
        
        # Forward-looking insights
        fls_ratio = features.get('forward_looking_ratio', 0)
        if fls_ratio > 0.1:
            insights.append("High frequency of forward-looking statements")
        
        # Readability insights
        flesch_score = features.get('flesch_reading_ease', 0)
        if flesch_score < 30:
            insights.append("Complex language suggests detailed technical discussion")
        elif flesch_score > 80:
            insights.append("Simple language suggests clear communication")
        
        return insights
