"""
NLP Sentiment Analyzer for market news and social media.
"""

import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import asyncio
import aiohttp
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


class SentimentAnalyzer:
    """
    Multi-model sentiment analyzer for financial news and text.
    Uses FinBERT for financial-specific sentiment and VADER for general sentiment.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        use_finbert: bool = True,
        use_vader: bool = True,
        device: Optional[str] = None,
        cache_size: int = 1000
    ):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name for financial sentiment
            use_finbert: Whether to use FinBERT
            use_vader: Whether to use VADER
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_size: Size of sentiment cache
        """
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE
        self.use_vader = use_vader and VADER_AVAILABLE

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TRANSFORMERS_AVAILABLE else 'cpu'
        else:
            self.device = device

        # Initialize FinBERT
        if self.use_finbert:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.finbert_model.to(self.device)
                self.finbert_model.eval()
                logger.info(f"FinBERT loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}")
                self.use_finbert = False

        # Initialize VADER
        if self.use_vader:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")

        # Sentiment cache
        self._cache: Dict[str, Dict] = {}
        self.cache_size = cache_size

        # News sources
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.investing.com/rss/news.rss",
        ]

        # Financial keywords for relevance scoring
        self.financial_keywords = {
            'bullish': ['buy', 'bullish', 'long', 'upgrade', 'growth', 'profit', 'surge', 'rally', 'gain'],
            'bearish': ['sell', 'bearish', 'short', 'downgrade', 'loss', 'decline', 'crash', 'drop', 'fall'],
            'neutral': ['hold', 'neutral', 'stable', 'unchanged', 'steady']
        }

    def analyze_text(
        self,
        text: str,
        use_cache: bool = True
    ) -> Dict[str, Union[float, str]]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze
            use_cache: Whether to use cached results

        Returns:
            Sentiment analysis results
        """
        # Check cache
        if use_cache and text in self._cache:
            return self._cache[text]

        results = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Clean text
        cleaned_text = self._clean_text(text)

        # FinBERT analysis
        if self.use_finbert:
            finbert_result = self._analyze_finbert(cleaned_text)
            results.update({
                'finbert_sentiment': finbert_result['sentiment'],
                'finbert_score': finbert_result['score'],
                'finbert_probabilities': finbert_result['probabilities']
            })

        # VADER analysis
        if self.use_vader:
            vader_result = self._analyze_vader(cleaned_text)
            results.update({
                'vader_compound': vader_result['compound'],
                'vader_positive': vader_result['pos'],
                'vader_negative': vader_result['neg'],
                'vader_neutral': vader_result['neu']
            })

        # Combined sentiment
        results['combined_sentiment'] = self._combine_sentiments(results)
        results['confidence'] = self._calculate_confidence(results)

        # Keyword analysis
        results['keyword_signals'] = self._analyze_keywords(cleaned_text)

        # Cache result
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[text] = results

        return results

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _analyze_finbert(self, text: str) -> Dict:
        """Analyze sentiment using FinBERT."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]

            labels = ['negative', 'neutral', 'positive']
            probs = probabilities.cpu().numpy()
            sentiment_idx = np.argmax(probs)

            return {
                'sentiment': labels[sentiment_idx],
                'score': float(probs[sentiment_idx]),
                'probabilities': {
                    labels[i]: float(probs[i]) for i in range(len(labels))
                }
            }
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'probabilities': {}}

    def _analyze_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'pos': scores['pos'],
                'neg': scores['neg'],
                'neu': scores['neu']
            }
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

    def _combine_sentiments(self, results: Dict) -> str:
        """Combine FinBERT and VADER sentiments."""
        score = 0.0
        weight = 0.0

        if 'finbert_score' in results:
            if results['finbert_sentiment'] == 'positive':
                score += results['finbert_score'] * 0.6
            elif results['finbert_sentiment'] == 'negative':
                score -= results['finbert_score'] * 0.6
            weight += 0.6

        if 'vader_compound' in results:
            score += results['vader_compound'] * 0.4
            weight += 0.4

        if weight == 0:
            return 'neutral'

        final_score = score / weight

        if final_score > 0.2:
            return 'bullish'
        elif final_score < -0.2:
            return 'bearish'
        return 'neutral'

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence in the sentiment analysis."""
        confidence = 0.0
        count = 0

        if 'finbert_score' in results:
            confidence += results['finbert_score']
            count += 1

        if 'vader_compound' in results:
            confidence += abs(results['vader_compound'])
            count += 1

        return confidence / count if count > 0 else 0.0

    def _analyze_keywords(self, text: str) -> Dict[str, int]:
        """Analyze financial keywords in text."""
        text_lower = text.lower()
        signals = {sentiment: 0 for sentiment in self.financial_keywords}

        for sentiment, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    signals[sentiment] += 1

        return signals

    def analyze_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            use_cache: Whether to use cached results

        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_text(text, use_cache) for text in texts]

    async def fetch_news(
        self,
        symbol: Optional[str] = None,
        max_items: int = 20
    ) -> List[Dict]:
        """
        Fetch news from RSS feeds asynchronously.

        Args:
            symbol: Optional symbol to filter news
            max_items: Maximum news items to return

        Returns:
            List of news items with sentiment
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available")
            return []

        news_items = []

        for source in self.news_sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:max_items]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    text = f"{title}. {summary}"

                    # Filter by symbol if provided
                    if symbol and symbol.upper() not in text.upper():
                        continue

                    sentiment = self.analyze_text(text)

                    news_items.append({
                        'title': title,
                        'summary': summary[:200],
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source,
                        'sentiment': sentiment
                    })
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {e}")

        return news_items[:max_items]

    def get_market_sentiment(
        self,
        news_items: List[Dict]
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate overall market sentiment from news items.

        Args:
            news_items: List of news items with sentiment

        Returns:
            Aggregated market sentiment
        """
        if not news_items:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }

        bullish = 0
        bearish = 0
        neutral = 0
        total_confidence = 0.0

        for item in news_items:
            sentiment = item.get('sentiment', {})
            combined = sentiment.get('combined_sentiment', 'neutral')
            confidence = sentiment.get('confidence', 0.0)

            if combined == 'bullish':
                bullish += 1
            elif combined == 'bearish':
                bearish += 1
            else:
                neutral += 1

            total_confidence += confidence

        total = len(news_items)
        avg_confidence = total_confidence / total

        # Calculate sentiment score
        score = (bullish - bearish) / total

        if score > 0.2:
            overall = 'bullish'
        elif score < -0.2:
            overall = 'bearish'
        else:
            overall = 'neutral'

        return {
            'sentiment': overall,
            'score': score,
            'confidence': avg_confidence,
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'total_news': total
        }

    def get_trading_signal(
        self,
        market_sentiment: Dict,
        threshold: float = 0.3
    ) -> Dict[str, Union[str, float]]:
        """
        Generate trading signal from market sentiment.

        Args:
            market_sentiment: Market sentiment analysis
            threshold: Threshold for generating signals

        Returns:
            Trading signal
        """
        score = market_sentiment.get('score', 0)
        confidence = market_sentiment.get('confidence', 0)

        if abs(score) < threshold or confidence < 0.3:
            return {
                'signal': 'hold',
                'strength': 0.0,
                'reason': 'Low confidence or weak signal'
            }

        if score > threshold:
            return {
                'signal': 'buy',
                'strength': min(1.0, score * confidence),
                'reason': f"Bullish sentiment: {market_sentiment['bullish_count']} positive news"
            }
        elif score < -threshold:
            return {
                'signal': 'sell',
                'strength': min(1.0, abs(score) * confidence),
                'reason': f"Bearish sentiment: {market_sentiment['bearish_count']} negative news"
            }

        return {
            'signal': 'hold',
            'strength': 0.0,
            'reason': 'Mixed signals'
        }

    def save(self, filepath: Union[str, Path]):
        """Save analyzer state (cache only, model is loaded from HF)."""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump({
                'cache_size': self.cache_size,
                'news_sources': self.news_sources
            }, f)
        logger.info(f"Sentiment analyzer config saved to {filepath}")

    def clear_cache(self):
        """Clear sentiment cache."""
        self._cache.clear()
        logger.info("Sentiment cache cleared")
