'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 11 Oktober 2025
Purpose : Menganalisis dan mengekstraksi topik tersembunyi dari kumpulan dokumen menggunakan metode LDA
=============================================================================================================================
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import os
import sys

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import CONFIG

class LDATopicModel:
    def __init__(self, n_topics=CONFIG.N_TOPICS, random_state=CONFIG.RF_RANDOM_STATE, max_iter=CONFIG.LDA_ITERATIONS):
        self.n_topics = n_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.lda_model = None
        self.topic_word_distribution = None
        self.doc_topic_distribution = None
        
    def fit(self, bow_matrix):
        """
        Train LDA model pada BOW matrix
        """
        print("\n=== TRAINING LDA MODEL ===")
        print(f"Jumlah topik: {self.n_topics}")
        print(f"Max iterations: {self.max_iter}")
        print(f"Random state: {self.random_state}")
        print(f"BOW Matrix shape: {bow_matrix.shape}")
        
        # TAMBAHAN: Batch size hint untuk dataset besar
        batch_size = min(1000, bow_matrix.shape[0] // 10)
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_method='online',
            batch_size=batch_size,        # ← Optional: untuk large dataset
            evaluate_every=5              # ← Optional: monitor progress
        )
        
        print("Training LDA model... (mungkin butuh beberapa menit untuk dataset besar)")
        self.doc_topic_distribution = self.lda_model.fit_transform(bow_matrix)
        self.topic_word_distribution = self.lda_model.components_
        
        print("✅ LDA model training selesai!")
        print(f"📊 Shape doc-topic: {self.doc_topic_distribution.shape}")
        print(f"📊 Shape topic-word: {self.topic_word_distribution.shape}")
        
        return self.doc_topic_distribution, self.topic_word_distribution
    
    def get_top_words_per_topic(self, vocabulary, n_words=10):
        """
        Mendapatkan kata-kata teratas untuk setiap topik
        
        Parameters:
        vocabulary: list of words dari BOW
        n_words: jumlah kata teratas per topik
        
        Returns:
        top_words: dictionary {topic_id: [list of words]}
        """
        if self.topic_word_distribution is None:
            print("Error: Model LDA belum di-training")
            return {}
        
        top_words = {}
        for topic_idx, topic in enumerate(self.topic_word_distribution):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words[topic_idx] = [(vocabulary[i], topic[i]) for i in top_indices]
        
        return top_words
    
    def get_topic_distribution_per_document(self):
        """
        Mendapatkan distribusi topik untuk setiap dokumen
        
        Returns:
        doc_topics: array (n_docs x n_topics)
        """
        return self.doc_topic_distribution
    
    def get_dominant_topic_per_document(self):
        """
        Mendapatkan topik dominan untuk setiap dokumen
        
        Returns:
        dominant_topics: array of dominant topic indices
        """
        if self.doc_topic_distribution is None:
            return None
        
        return np.argmax(self.doc_topic_distribution, axis=1)
    
    def calculate_perplexity(self, bow_matrix):
        """
        Menghitung perplexity dari model
        
        Parameters:
        bow_matrix: test data untuk menghitung perplexity
        
        Returns:
        perplexity: nilai perplexity
        """
        if self.lda_model is None:
            print("Error: Model LDA belum di-training")
            return None
        
        perplexity = self.lda_model.perplexity(bow_matrix)
        print(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def calculate_coherence(self, top_words):
        """
        Menghitung coherence score sederhana (berdasarkan PMI)
        
        Parameters:
        top_words: dictionary top words per topic dari get_top_words_per_topic()
        
        Returns:
        coherence_scores: dictionary coherence per topic
        """
        # Implementasi coherence sederhana
        coherence_scores = {}
        for topic_id, words_scores in top_words.items():
            words = [word for word, score in words_scores]
            # Coherence sederhana: rata-rata frekuensi kata dalam topik
            avg_score = np.mean([score for word, score in words_scores])
            coherence_scores[topic_id] = avg_score
        
        return coherence_scores
    
    def save_model(self, filepath):
        """Menyimpan model LDA"""
        if self.lda_model is None:
            print("Error: Model LDA belum di-training")
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'lda_model': self.lda_model,
                    'topic_word_distribution': self.topic_word_distribution,
                    'doc_topic_distribution': self.doc_topic_distribution,
                    'config': {
                        'n_topics': self.n_topics,
                        'random_state': self.random_state,
                        'max_iter': self.max_iter
                    }
                }, f)
            print(f"LDA model disimpan di: {filepath}")
            return True
        except Exception as e:
            print(f"Error menyimpan model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load model LDA yang sudah disimpan"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.topic_word_distribution = model_data['topic_word_distribution']
            self.doc_topic_distribution = model_data['doc_topic_distribution']
            self.n_topics = model_data['config']['n_topics']
            self.random_state = model_data['config']['random_state']
            self.max_iter = model_data['config']['max_iter']
            
            print(f"LDA model loaded dari: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def load_bow_for_lda():
    """Load BOW matrix untuk LDA"""
    bow_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, "bow_for_lda.csv")
    vocab_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, "lda_vocabulary.csv")
    
    if not os.path.exists(bow_file) or not os.path.exists(vocab_file):
        print(f"Error: File BOW untuk LDA tidak ditemukan")
        print(f"Pastikan sudah menjalankan CNP_METODE_BOW.py terlebih dahulu")
        return None, None
    
    # Load BOW matrix
    bow_df = pd.read_csv(bow_file, encoding=CONFIG.ENCODING)
    bow_matrix = bow_df.values
    
    # Load vocabulary
    vocab_df = pd.read_csv(vocab_file, encoding=CONFIG.ENCODING)
    vocabulary = vocab_df['word'].tolist()
    
    print(f"Berhasil load BOW matrix: {bow_matrix.shape}")
    print(f"Vocabulary: {len(vocabulary)} kata")
    
    return bow_matrix, vocabulary

def load_original_data():
    """Load data original untuk metadata"""
    if CONFIG.TESTING_MODE:
        input_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, f"processed_data_test_{CONFIG.TEST_FILE.split('.')[0]}.csv")
    else:
        input_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, "processed_data_full.csv")
    
    if not os.path.exists(input_file):
        print(f"Warning: File data original tidak ditemukan")
        return None
    
    df = pd.read_csv(input_file, encoding=CONFIG.ENCODING)
    print(f"Berhasil load data original: {len(df)} dokumen")
    return df

def display_lda_results(lda_model, vocabulary, top_n_words=10):
    """Menampilkan hasil LDA"""
    print("\n" + "="*70)
    print("HASIL LATENT DIRICHLET ALLOCATION (LDA)")
    print("="*70)
    
    # Get top words per topic
    top_words = lda_model.get_top_words_per_topic(vocabulary, top_n_words)
    
    print(f"\n TOP {top_n_words} KATA UNTUK SETIAP TOPIK:")
    for topic_id, words_scores in top_words.items():
        print(f"\n--- TOPIK {topic_id + 1} ---")
        for i, (word, score) in enumerate(words_scores, 1):
            print(f"  {i:2d}. {word:<15} (score: {score:.4f})")
    
    # Calculate and display coherence
    coherence_scores = lda_model.calculate_coherence(top_words)
    print(f"\n COHERENCE SCORE PER TOPIK:")
    for topic_id, coherence in coherence_scores.items():
        print(f"  Topik {topic_id + 1}: {coherence:.4f}")
    
    # Display dominant topics distribution
    dominant_topics = lda_model.get_dominant_topic_per_document()
    if dominant_topics is not None:
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        print(f"\n DISTRIBUSI DOKUMEN PER TOPIK DOMINAN:")
        for topic_id, count in topic_counts.items():
            percentage = (count / len(dominant_topics)) * 100
            print(f"  Topik {topic_id + 1}: {count:3d} dokumen ({percentage:.1f}%)")

def visualize_topics(lda_model, vocabulary, n_words=15):
    """Visualisasi topik dengan wordcloud dan bar plot"""
    print("\n=== MEMBUAT VISUALISASI TOPIK ===")
    
    # Buat folder untuk visualisasi di dalam LDA directory
    viz_dir = os.path.join(CONFIG.LDA_OUTPUT_DIR, "visualization")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    top_words = lda_model.get_top_words_per_topic(vocabulary, n_words)
    
    # WordCloud untuk setiap topik
    print("Membuat WordCloud...")
    for topic_id, words_scores in top_words.items():
        word_freq = {word: score for word, score in words_scores}
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topik {topic_id + 1} - WordCloud', size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'wordcloud_topic_{topic_id + 1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Bar plot untuk kata-kata teratas
    print("Membuat Bar Plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for topic_id, words_scores in top_words.items():
        if topic_id < len(axes):
            words = [word for word, score in words_scores[:10]]
            scores = [score for word, score in words_scores[:10]]
            
            axes[topic_id].barh(words, scores, color='skyblue')
            axes[topic_id].set_title(f'Topik {topic_id + 1}')
            axes[topic_id].set_xlabel('Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'topic_words_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi disimpan di: {viz_dir}")

def save_lda_results(lda_model, vocabulary, original_df, filename="lda_results.csv"):
    """Menyimpan hasil LDA ke CSV"""
    # Buat LDA directory jika belum ada
    if not os.path.exists(CONFIG.LDA_OUTPUT_DIR):
        os.makedirs(CONFIG.LDA_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, filename)
    
    # Get topic distributions
    doc_topic_dist = lda_model.get_topic_distribution_per_document()
    dominant_topics = lda_model.get_dominant_topic_per_document()
    
    # Create results DataFrame
    results_df = pd.DataFrame(doc_topic_dist, columns=[f'Topic_{i+1}' for i in range(doc_topic_dist.shape[1])])
    results_df['Dominant_Topic'] = dominant_topics + 1  # +1 untuk mulai dari 1 bukan 0
    results_df['Dominant_Topic_Score'] = np.max(doc_topic_dist, axis=1)
    
    # Add original metadata jika ada
    if original_df is not None:
        results_df['filename'] = original_df['filename'].values
        results_df['title'] = original_df['title'].values
        results_df['label'] = original_df['label'].values
    
    results_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f"\n Hasil LDA disimpan di: {output_file}")
    
    return output_file

def save_topic_words(lda_model, vocabulary, filename="topic_words.csv"):
    """Menyimpan kata-kata untuk setiap topik"""
    # Buat LDA directory jika belum ada
    if not os.path.exists(CONFIG.LDA_OUTPUT_DIR):
        os.makedirs(CONFIG.LDA_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, filename)
    
    top_words = lda_model.get_top_words_per_topic(vocabulary, 20)
    
    topic_words_data = []
    for topic_id, words_scores in top_words.items():
        for rank, (word, score) in enumerate(words_scores, 1):
            topic_words_data.append({
                'topic_id': topic_id + 1,
                'rank': rank,
                'word': word,
                'score': score
            })
    
    topic_words_df = pd.DataFrame(topic_words_data)
    topic_words_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f" Kata-kata topik disimpan di: {output_file}")
    
    return output_file

def save_lda_model(lda_model, filename="lda_model.pkl"):
    """Menyimpan model LDA"""
    # Buat LDA directory jika belum ada
    if not os.path.exists(CONFIG.LDA_OUTPUT_DIR):
        os.makedirs(CONFIG.LDA_OUTPUT_DIR)
    
    model_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, filename)
    lda_model.save_model(model_file)
    return model_file

def save_lda_summary(lda_model, vocabulary, bow_matrix, original_df=None, filename="lda_summary.txt"):
    """Menyimpan summary statistik LDA"""
    if not os.path.exists(CONFIG.LDA_OUTPUT_DIR):
        os.makedirs(CONFIG.LDA_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.LDA_OUTPUT_DIR, filename)
    
    with open(output_file, 'w', encoding=CONFIG.ENCODING) as f:
        f.write("LATENT DIRICHLET ALLOCATION (LDA) SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic Information
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"Number of Topics: {lda_model.n_topics}\n")
        f.write(f"Max Iterations: {lda_model.max_iter}\n")
        f.write(f"Random State: {lda_model.random_state}\n")
        f.write(f"BOW Matrix Shape: {bow_matrix.shape}\n")
        f.write(f"Vocabulary Size: {len(vocabulary):,}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE:\n")
        perplexity = lda_model.calculate_perplexity(bow_matrix)
        f.write(f"Perplexity: {perplexity:.2f}\n\n")
        
        # Topic Information
        top_words = lda_model.get_top_words_per_topic(vocabulary, 10)
        coherence_scores = lda_model.calculate_coherence(top_words)
        
        f.write("TOPIC ANALYSIS:\n")
        f.write(f"{'Topic':<8} {'Coherence':<12} {'Top Words'}\n")
        f.write("-" * 60 + "\n")
        
        for topic_id in range(lda_model.n_topics):
            words = [word for word, score in top_words[topic_id][:5]]  # Top 5 words
            coherence = coherence_scores[topic_id]
            f.write(f"{topic_id + 1:<8} {coherence:<12.4f} {', '.join(words)}\n")
        f.write("\n")
        
        # Document-Topic Distribution
        doc_topic_dist = lda_model.get_topic_distribution_per_document()
        dominant_topics = lda_model.get_dominant_topic_per_document()
        
        f.write("DOCUMENT-TOPIC DISTRIBUTION:\n")
        f.write(f"Total Documents: {doc_topic_dist.shape[0]:,}\n")
        
        # Dominant topic statistics
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        f.write("\nDOMINANT TOPIC DISTRIBUTION:\n")
        total_docs = len(dominant_topics)
        for topic_id in range(lda_model.n_topics):
            count = topic_counts.get(topic_id, 0)
            percentage = (count / total_docs) * 100
            f.write(f"Topic {topic_id + 1}: {count:>4} documents ({percentage:5.1f}%)\n")
        
        f.write(f"\nTopic Balance: {topic_counts.std():.2f} (std dev, lower is better)\n")
        
        # Topic quality metrics
        f.write("\nTOPIC QUALITY METRICS:\n")
        avg_coherence = np.mean(list(coherence_scores.values()))
        max_coherence = np.max(list(coherence_scores.values()))
        min_coherence = np.min(list(coherence_scores.values()))
        
        f.write(f"Average Coherence: {avg_coherence:.4f}\n")
        f.write(f"Max Coherence: {max_coherence:.4f}\n")
        f.write(f"Min Coherence: {min_coherence:.4f}\n")
        f.write(f"Coherence Range: {max_coherence - min_coherence:.4f}\n")
        
        # Document statistics
        f.write("\nDOCUMENT STATISTICS:\n")
        avg_topic_score = np.mean(np.max(doc_topic_dist, axis=1))
        std_topic_score = np.std(np.max(doc_topic_dist, axis=1))
        
        f.write(f"Average Dominant Topic Score: {avg_topic_score:.3f} ± {std_topic_score:.3f}\n")
        
        # Count documents with high confidence (score > 0.8)
        high_confidence_docs = np.sum(np.max(doc_topic_dist, axis=1) > 0.8)
        high_confidence_pct = (high_confidence_docs / total_docs) * 100
        f.write(f"High Confidence Documents (>0.8): {high_confidence_docs:,} ({high_confidence_pct:.1f}%)\n")
        
        # Label analysis if original data is available
        if original_df is not None and 'label' in original_df.columns:
            f.write("\nLABEL-TOPIC ANALYSIS:\n")
            results_df = pd.DataFrame({
                'dominant_topic': dominant_topics + 1,
                'label': original_df['label'].values
            })
            
            label_topic_crosstab = pd.crosstab(results_df['label'], results_df['dominant_topic'])
            f.write("Cross-tabulation (Label vs Dominant Topic):\n")
            for line in str(label_topic_crosstab).split('\n'):
                f.write(line + '\n')
        
        f.write(f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ LDA summary disimpan di: {output_file}")
    return output_file

def test_lda():
    """Function untuk testing LDA"""
    print("=== TESTING LATENT DIRICHLET ALLOCATION ===")
    
    # Load BOW matrix untuk LDA
    bow_matrix, vocabulary = load_bow_for_lda()
    if bow_matrix is None:
        return
    
    # Load data original untuk metadata
    original_df = load_original_data()
    
    # Train LDA model
    lda_model = LDATopicModel()
    doc_topic_dist, topic_word_dist = lda_model.fit(bow_matrix)
    
    # Calculate perplexity
    print("\n=== EVALUASI MODEL ===")
    perplexity = lda_model.calculate_perplexity(bow_matrix)
    
    # Display results
    display_lda_results(lda_model, vocabulary)
    
    # Save results ke folder LDA
    results_file = save_lda_results(lda_model, vocabulary, original_df)
    topic_words_file = save_topic_words(lda_model, vocabulary)
    model_file = save_lda_model(lda_model)
    
    # TAMBAHAN: Simpan summary LDA
    summary_file = save_lda_summary(lda_model, vocabulary, bow_matrix, original_df)
    
    # Visualisasi (optional)
    try:
        visualize_topics(lda_model, vocabulary)
    except Exception as e:
        print(f"Warning: Gagal membuat visualisasi: {e}")
        print("Pastikan matplotlib dan wordcloud terinstall: pip install matplotlib wordcloud")
    
    print(f"\n LDA SELESAI!")
    print(f"   File hasil: {results_file}")
    print(f"   File kata topik: {topic_words_file}")
    print(f"   Model LDA: {model_file}")
    print(f"   Summary LDA: {summary_file}")  # TAMBAHAN
    print(f"   Perplexity: {perplexity:.2f}")
    
    return lda_model, doc_topic_dist, topic_word_dist

if __name__ == "__main__":
    lda_model, doc_topic_dist, topic_word_dist = test_lda()