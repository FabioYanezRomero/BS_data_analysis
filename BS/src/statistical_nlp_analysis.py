"""
statistical_nlp_analysis.py

Performs per-intent NLP data-quality analysis for a Kore.ai chatbot:
 - Sentence/word length statistics (mean, percentiles, kurtosis, 99th percentile)
 - Lexical richness (vocabulary size, type-token ratio)
 - Average syllables per word (readability proxy)
 - Outlier detection (counts and ratios beyond 99th percentile)
 - Train/Test set consistency metrics (KL, JS divergences and Earth Mover's Distance; TTR difference)
 - Generates combined boxplots comparing train vs. test distributions across intents

Outputs:
 - utterances_stats.json, tests_stats.json: summary metrics per intent
 - divergence_stats.json: consistency metrics between train/test
 - sentence_length_boxplot.png, word_length_boxplot.png: visual comparisons
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, entropy, wasserstein_distance
from nltk.tokenize import RegexpTokenizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NLPAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the NLP analyzer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.embedder = SentenceTransformer(model_name)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.percentiles = [10, 25, 50, 75, 90]
    
    def count_syllables(self, word):
        """
        Return number of syllables in a Spanish word by counting vowel groups.
        """
        # Spanish vowel groups
        return len(re.findall(r'[aeiouáéíóúü]+', word.lower()))
    
    def analyze_intent_json(self, json_path):
        """
        Analyze an intent JSON file containing intent-to-utterances mappings.
        
        Args:
            json_path: Path to the JSON file containing intent data
            
        Returns:
            tuple: (results, distributions) where results contains statistics and
                  distributions contains raw data distributions
        """
        # Load intent-to-utterances mapping from JSON
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
            
        # Containers for summary metrics and raw distributions
        results = {}
        distributions = {}
        
        for intent, utterances in data.items():
            # Filter out empty or non-string utterances
            texts = [u for u in utterances if isinstance(u, str) and u.strip()]
            
            # Compute sentence lengths (number of tokens per utterance)
            sentence_lengths = [len(self.tokenizer.tokenize(t.lower())) for t in texts]
            
            # Flatten all word tokens across utterances
            all_words = [w for t in texts for w in self.tokenizer.tokenize(t.lower())]
            
            # Compute word lengths (number of characters per token)
            word_lengths = [len(w) for w in all_words]
            
            # Lexical richness: vocabulary size and type-token ratio
            vocab = set(all_words)
            vocab_size = len(vocab)
            ttr = vocab_size / len(all_words) if all_words else 0.0
            
            # Readability proxy: average syllables per word
            syllable_counts = [self.count_syllables(w) for w in all_words]
            avg_syllables_per_word = float(np.mean(syllable_counts)) if syllable_counts else 0.0
            
            # Embedding-based cohesion: semantic tightness per intent
            if texts:
                embeddings = self.embedder.encode(texts, convert_to_numpy=True)
                centroid = np.mean(embeddings, axis=0)
                sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
                embedding_cohesion = float(np.mean(sims))
            else:
                centroid = None
                embedding_cohesion = None
            
            # Outlier detection: count utterances/words beyond the 99th percentile
            sent_threshold = float(np.percentile(sentence_lengths, 99)) if sentence_lengths else 0.0
            num_sentence_outliers = int(sum(l > sent_threshold for l in sentence_lengths)) if sentence_lengths else 0
            sentence_outlier_ratio = float(num_sentence_outliers / len(texts)) if texts else 0.0
            
            word_threshold = float(np.percentile(word_lengths, 99)) if word_lengths else 0.0
            num_word_outliers = int(sum(l > word_threshold for l in word_lengths)) if word_lengths else 0
            word_outlier_ratio = float(num_word_outliers / len(all_words)) if all_words else 0.0
            
            # Metrics
            stats = {
                'num_examples': len(texts),
                'vocab_size': vocab_size,
                'type_token_ratio': ttr,  # lexical diversity measure
                'avg_sentence_length': float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
                'sentence_length_percentiles': {str(p): float(np.percentile(sentence_lengths, p)) for p in self.percentiles} if sentence_lengths else {},
                'sentence_length_kurtosis': float(kurtosis(sentence_lengths)) if len(sentence_lengths) > 3 else None,
                'sentence_length_99_percentile': float(np.percentile(sentence_lengths, 99)) if sentence_lengths else None,
                'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0.0,
                'word_length_percentiles': {str(p): float(np.percentile(word_lengths, p)) for p in self.percentiles} if word_lengths else {},
                'word_length_kurtosis': float(kurtosis(word_lengths)) if len(word_lengths) > 3 else None,
                'avg_syllables_per_word': avg_syllables_per_word,
                'embedding_cohesion': embedding_cohesion,
                'num_sentence_outliers': num_sentence_outliers,
                'sentence_outlier_ratio': sentence_outlier_ratio,
                'num_word_outliers': num_word_outliers,
                'word_outlier_ratio': word_outlier_ratio
            }
            
            results[intent] = stats
            distributions[intent] = {
                'sentence_lengths': sentence_lengths,
                'word_lengths': word_lengths,
                'type_token_ratio': ttr,
                'embedding_centroid': centroid
            }
            
        return results, distributions
    
    def compute_divergence(self, train_dist, test_dist):
        """
        Compute divergence metrics between training and test distributions.
        
        Args:
            train_dist: Training data distributions
            test_dist: Test data distributions
            
        Returns:
            dict: Divergence metrics per intent
        """
        divergence = {}
        
        for intent in set(train_dist) | set(test_dist):
            # Sentence lengths
            sl_train = train_dist.get(intent, {}).get('sentence_lengths', [])
            sl_test = test_dist.get(intent, {}).get('sentence_lengths', [])
            
            # Earth Mover's Distance (EMD)
            EMD_sl = wasserstein_distance(sl_train, sl_test)
            
            lengths = sorted(set(sl_train) | set(sl_test))
            train_counts = np.array([sl_train.count(l) for l in lengths], dtype=float)
            test_counts = np.array([sl_test.count(l) for l in lengths], dtype=float)
            
            eps = 1e-9
            P = (train_counts + eps) / (train_counts.sum() + eps * len(lengths))
            Q = (test_counts + eps) / (test_counts.sum() + eps * len(lengths))
            
            KL_sl = float(entropy(P, Q))
            M = 0.5 * (P + Q)
            JS_sl = float(0.5 * entropy(P, M) + 0.5 * entropy(Q, M))
            
            # Word lengths
            wl_train = train_dist.get(intent, {}).get('word_lengths', [])
            wl_test = test_dist.get(intent, {}).get('word_lengths', [])
            
            # Earth Mover's Distance for word lengths
            EMD_w = wasserstein_distance(wl_train, wl_test)
            
            lengths_w = sorted(set(wl_train) | set(wl_test))
            train_counts_w = np.array([wl_train.count(l) for l in lengths_w], dtype=float)
            test_counts_w = np.array([wl_test.count(l) for l in lengths_w], dtype=float)
            
            P_w = (train_counts_w + eps) / (train_counts_w.sum() + eps * len(lengths_w))
            Q_w = (test_counts_w + eps) / (test_counts_w.sum() + eps * len(lengths_w))
            
            KL_w = float(entropy(P_w, Q_w))
            M_w = 0.5 * (P_w + Q_w)
            JS_w = float(0.5 * entropy(P_w, M_w) + 0.5 * entropy(Q_w, M_w))
            
            # TTR difference
            ttr_train = train_dist.get(intent, {}).get('type_token_ratio', 0.0)
            ttr_test = test_dist.get(intent, {}).get('type_token_ratio', 0.0)
            ttr_diff = abs(ttr_train - ttr_test)
            
            divergence[intent] = {
                'sentence_length_KL_divergence': KL_sl,
                'sentence_length_JS_divergence': JS_sl,
                'sentence_length_EMD': EMD_sl,
                'word_length_KL_divergence': KL_w,
                'word_length_JS_divergence': JS_w,
                'word_length_EMD': EMD_w,
                'type_token_ratio_diff': ttr_diff
            }
            
            # Embedding centroid cosine similarity: measures semantic coherence shift between train and test
            c_train = train_dist.get(intent, {}).get('embedding_centroid')
            c_test = test_dist.get(intent, {}).get('embedding_centroid')
            
            if c_train is not None and c_test is not None:
                csim = float(cosine_similarity(c_train.reshape(1, -1), c_test.reshape(1, -1))[0][0])
            else:
                csim = None
                
            divergence[intent]['embedding_centroid_cosine_similarity'] = csim
            
        return divergence
    
    def create_visualizations(self, train_dist, test_dist, output_dir):
        """
        Create visualizations comparing train and test distributions.
        
        Args:
            train_dist: Training data distributions
            test_dist: Test data distributions
            output_dir: Directory to save the plots
        """
        # Sentence lengths boxplot
        df_sl = pd.DataFrame([
            {'intent': intent, 'dataset': 'train', 'length': l}
            for intent, lengths in train_dist.items() for l in lengths.get('sentence_lengths', [])
        ] + [
            {'intent': intent, 'dataset': 'test', 'length': l}
            for intent, lengths in test_dist.items() for l in lengths.get('sentence_lengths', [])
        ])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='intent', y='length', hue='dataset', data=df_sl)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Sentence length (words)')
        plt.tight_layout()
        plt.savefig(str(output_dir / 'sentence_length_boxplot.png'))
        plt.close()
        
        # Word lengths boxplot
        df_wl = pd.DataFrame([
            {'intent': intent, 'dataset': 'train', 'length': l}
            for intent, lengths in train_dist.items() for l in lengths.get('word_lengths', [])
        ] + [
            {'intent': intent, 'dataset': 'test', 'length': l}
            for intent, lengths in test_dist.items() for l in lengths.get('word_lengths', [])
        ])
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='intent', y='length', hue='dataset', data=df_wl)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Word length (characters)')
        plt.tight_layout()
        plt.savefig(str(output_dir / 'word_length_boxplot.png'))
        plt.close()


def main():
    """Main function to run the NLP analysis"""
    # Set up paths
    base = Path(__file__).parent.parent / 'utterances_tests'
    train_path = base / 'utterances.json'
    test_path = base / 'tests.json'
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Analyze train and test data
    print("Analyzing training data...")
    train_stats, train_dist = analyzer.analyze_intent_json(train_path)
    
    print("Analyzing test data...")
    test_stats, test_dist = analyzer.analyze_intent_json(test_path)
    
    # Save separate stats
    with open(base / 'utterances_stats.json', 'w', encoding='utf-8') as f:
        json.dump(train_stats, f, ensure_ascii=False, indent=2)
    with open(base / 'tests_stats.json', 'w', encoding='utf-8') as f:
        json.dump(test_stats, f, ensure_ascii=False, indent=2)
    print('Saved train/test stats JSON.')
    
    # Compute and save divergence metrics
    print("Computing divergence metrics...")
    divergence = analyzer.compute_divergence(train_dist, test_dist)
    with open(base / 'divergence_stats.json', 'w', encoding='utf-8') as f:
        json.dump(divergence, f, ensure_ascii=False, indent=2)
    print('Saved divergence_stats.json')
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(train_dist, test_dist, base)
    print("Saved boxplot visualizations.")


if __name__ == '__main__':
    main()
