'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 10 Oktober 2025
Purpose : Mengubah teks menjadi vektor numerik menggunakan Bag-of-Words
=============================================================================================================================
'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import sys
from datetime import datetime

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import CONFIG

# Import preprocessing module
sys.path.append(os.path.dirname(__file__))
from CNP_PREPROCESSING import load_tokens_from_csv

class BOWTransformer:
    def __init__(self, max_features=CONFIG.MAX_FEATURES, min_df=CONFIG.MIN_DF, max_df=CONFIG.MAX_DF):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.vocabulary = None
        
    def create_bow_matrix(self, documents):
        """
        Membuat Bag-of-Words matrix dari list of tokens
        
        Parameters:
        documents: list of list of tokens -> [['token1', 'token2', ...], ...]
        
        Returns:
        bow_matrix: sparse matrix (n_docs x n_features)
        feature_names: list of vocabulary
        """
        print("\n" + "="*60)
        print("🎯 MEMBUAT BAG-OF-WORDS MATRIX")
        print("="*60)
        
        print(f"📊 Jumlah dokumen yang akan diproses: {len(documents):,}")
        
        # Convert list of tokens to list of strings untuk CountVectorizer
        print("🔄 Mengkonversi tokens ke string...")
        documents_as_strings = [' '.join(tokens) for tokens in documents]
        
        # Hitung statistik dasar
        total_tokens = sum(len(tokens) for tokens in documents)
        avg_tokens_per_doc = total_tokens / len(documents)
        print(f"📝 Total tokens: {total_tokens:,}")
        print(f"📝 Rata-rata tokens per dokumen: {avg_tokens_per_doc:.1f}")
        
        # Inisialisasi CountVectorizer dengan optimasi untuk dataset besar
        print("⚙️  Menginisialisasi CountVectorizer...")
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None,  # Karena sudah dihandle di preprocessing
            dtype=np.int32,   # Optimasi memory
            lowercase=False   # Karena sudah lowercase di preprocessing
        )
        
        # Fit dan transform documents
        print("🔧 Memproses BOW matrix...")
        start_time = datetime.now()
        bow_matrix = self.vectorizer.fit_transform(documents_as_strings)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        print(f"✅ BOW Matrix berhasil dibuat dalam {processing_time:.2f} detik")
        print(f"📊 Shape BOW Matrix: {bow_matrix.shape}")
        print(f"🔤 Jumlah fitur (vocabulary): {len(self.vocabulary):,}")
        print(f"📝 Jumlah dokumen: {bow_matrix.shape[0]:,}")
        
        return bow_matrix, self.vocabulary
    
    def get_feature_names(self):
        """Mendapatkan daftar feature names (vocabulary)"""
        return self.vocabulary if self.vocabulary is not None else []
    
    def get_matrix_stats(self, bow_matrix):
        """Mendapatkan statistik dari BOW matrix"""
        print("📈 Menghitung statistik BOW matrix...")
        
        # Gunakan method sparse untuk hemat memory
        total_elements = bow_matrix.shape[0] * bow_matrix.shape[1]
        sparsity = 1 - (bow_matrix.nnz / total_elements)
        
        # Hitung words per document menggunakan sparse operations
        words_per_doc = bow_matrix.sum(axis=1).A1  # Convert to 1D array
        
        stats = {
            'total_documents': bow_matrix.shape[0],
            'total_features': bow_matrix.shape[1],
            'total_non_zero_elements': bow_matrix.nnz,
            'sparsity': sparsity,
            'average_words_per_doc': words_per_doc.mean(),
            'max_words_in_doc': words_per_doc.max(),
            'min_words_in_doc': words_per_doc.min(),
            'std_words_per_doc': words_per_doc.std(),
            'memory_size_mb': (bow_matrix.data.nbytes + bow_matrix.indptr.nbytes + bow_matrix.indices.nbytes) / (1024 * 1024),
            'dense_size_gb': (bow_matrix.shape[0] * bow_matrix.shape[1] * 4) / (1024 ** 3)  # Estimated dense size
        }
        
        return stats
    
    def get_vocabulary_stats(self, bow_matrix):
        """Mendapatkan statistik vocabulary"""
        print("📊 Menganalisis distribusi vocabulary...")
        
        # Hitung frekuensi setiap kata
        word_frequencies = bow_matrix.sum(axis=0).A1
        
        stats = {
            'most_frequent_words': [],
            'least_frequent_words': [],
            'word_frequency_range': {
                'min': word_frequencies.min(),
                'max': word_frequencies.max(),
                'mean': word_frequencies.mean(),
                'median': np.median(word_frequencies)
            }
        }
        
        # Top 10 most frequent words
        top_indices = word_frequencies.argsort()[-10:][::-1]
        stats['most_frequent_words'] = [
            (self.vocabulary[idx], word_frequencies[idx]) 
            for idx in top_indices
        ]
        
        # Top 10 least frequent words (but appearing at least once)
        non_zero_indices = word_frequencies.nonzero()[0]
        if len(non_zero_indices) > 10:
            bottom_indices = word_frequencies[non_zero_indices].argsort()[:10]
            stats['least_frequent_words'] = [
                (self.vocabulary[non_zero_indices[idx]], word_frequencies[non_zero_indices[idx]]) 
                for idx in bottom_indices
            ]
        
        return stats
    
    def save_bow_model(self, filepath):
        """Menyimpan BOW model untuk penggunaan future"""
        if self.vectorizer is None:
            print("❌ Error: Model BOW belum dibuat")
            return False
            
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'vocabulary': self.vocabulary,
                'config': {
                    'max_features': self.max_features,
                    'min_df': self.min_df,
                    'max_df': self.max_df
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"💾 BOW model disimpan di: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error menyimpan model: {e}")
            return False
    
    def load_bow_model(self, filepath):
        """Load BOW model yang sudah disimpan"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.vocabulary = model_data['vocabulary']
            self.max_features = model_data['config']['max_features']
            self.min_df = model_data['config']['min_df']
            self.max_df = model_data['config']['max_df']
            
            print(f"✅ BOW model loaded dari: {filepath}")
            print(f"📅 Timestamp: {model_data.get('timestamp', 'Unknown')}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def load_processed_data():
    """Load data yang sudah diproses dari CSV"""
    if CONFIG.TESTING_MODE:
        input_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, f"processed_data_test_{CONFIG.TEST_FILE.split('.')[0]}.csv")
    else:
        input_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, "processed_data_full.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File {input_file} tidak ditemukan")
        print("💡 Jalankan CNP_PREPROCESSING.py terlebih dahulu")
        return None
    
    print(f"📂 Memuat data dari: {input_file}")
    df = load_tokens_from_csv(input_file)
    
    if df.empty:
        print("❌ Tidak ada data yang berhasil dimuat")
        return None
        
    print(f"✅ Berhasil load {len(df):,} dokumen")
    return df

def display_bow_results(bow_matrix, vocabulary, stats, vocab_stats, top_n=30):
    """Menampilkan hasil BOW"""
    print("\n" + "="*70)
    print("🎯 HASIL BAG-OF-WORDS TRANSFORMATION")
    print("="*70)
    
    print(f"\n📊 STATISTIK BOW MATRIX:")
    print(f"   • Total Dokumen          : {stats['total_documents']:,}")
    print(f"   • Total Fitur            : {stats['total_features']:,}")
    print(f"   • Elemen Non-Zero        : {stats['total_non_zero_elements']:,}")
    print(f"   • Sparsity               : {stats['sparsity']:.2%}")
    print(f"   • Rata-rata Kata/Dokumen : {stats['average_words_per_doc']:.1f} ± {stats['std_words_per_doc']:.1f}")
    print(f"   • Min-Max Kata/Dokumen   : {stats['min_words_in_doc']} - {stats['max_words_in_doc']}")
    print(f"   • Memory Usage (Sparse)  : {stats['memory_size_mb']:.1f} MB")
    print(f"   • Estimated Dense Size   : {stats['dense_size_gb']:.1f} GB")
    
    print(f"\n📈 STATISTIK VOCABULARY:")
    freq_range = vocab_stats['word_frequency_range']
    print(f"   • Frekuensi Kata Rata-rata : {freq_range['mean']:.1f}")
    print(f"   • Frekuensi Kata Median    : {freq_range['median']:.1f}")
    print(f"   • Range Frekuensi          : {freq_range['min']} - {freq_range['max']}")
    
    print(f"\n🏆 TOP {top_n} KATA PALING SERING MUNCUL:")
    word_frequencies = bow_matrix.sum(axis=0).A1
    top_indices = word_frequencies.argsort()[-top_n:][::-1]
    
    total_words = word_frequencies.sum()
    
    for i, idx in enumerate(top_indices, 1):
        frequency = word_frequencies[idx]
        percentage = (frequency / total_words) * 100
        doc_frequency = (bow_matrix[:, idx] > 0).sum()  # Jumlah dokumen yang mengandung kata ini
        doc_percentage = (doc_frequency / stats['total_documents']) * 100
        
        print(f"   {i:2d}. {vocabulary[idx]:<20} : {frequency:>6,} kali ({percentage:5.2f}%) - {doc_frequency:>4} dokumen ({doc_percentage:4.1f}%)")
    
    print(f"\n🔍 10 KATA PALING JARANG MUNCUL (tapi ada):")
    if vocab_stats['least_frequent_words']:
        for i, (word, freq) in enumerate(vocab_stats['least_frequent_words'][:10], 1):
            print(f"   {i:2d}. {word:<20} : {freq:>6} kali")
    
    # Tampilkan sample dokumen
    print(f"\n📄 SAMPLE DOKUMEN (3 pertama):")
    for i in range(min(3, bow_matrix.shape[0])):
        doc_vector = bow_matrix[i].toarray().flatten()
        non_zero_indices = doc_vector.nonzero()[0]
        
        print(f"   📝 Dokumen {i+1}:")
        print(f"      • Total kata unik  : {len(non_zero_indices)}")
        print(f"      • Total semua kata : {doc_vector.sum()}")
        if len(non_zero_indices) > 0:
            top_words_indices = non_zero_indices[doc_vector[non_zero_indices].argsort()[-5:][::-1]]
            top_words = [(vocabulary[idx], doc_vector[idx]) for idx in top_words_indices]
            print(f"      • 5 kata teratas   : {top_words}")

def save_bow_results(bow_matrix, vocabulary, df, filename="bow_results.csv"):
    """Menyimpan hasil BOW ke CSV"""
    # Buat folder output/BoW jika belum ada
    if not os.path.exists(CONFIG.BOW_OUTPUT_DIR):
        os.makedirs(CONFIG.BOW_OUTPUT_DIR)
        print(f"📁 Directory created: {CONFIG.BOW_OUTPUT_DIR}")
    
    output_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, filename)
    
    print(f"💾 Menyimpan hasil BOW ke CSV...")
    
    # Untuk dataset besar, simpan dalam chunks atau sample saja
    if bow_matrix.shape[0] > 10000:
        print("   ⚠️  Dataset besar, menyimpan sample 10,000 baris pertama...")
        sample_size = 10000
        bow_sample = bow_matrix[:sample_size]
        df_sample = df.head(sample_size)
    else:
        bow_sample = bow_matrix
        df_sample = df
    
    # Buat DataFrame dari BOW matrix
    bow_dense = bow_sample.toarray()
    bow_df = pd.DataFrame(bow_dense, columns=vocabulary)
    
    # Tambahkan metadata
    bow_df['filename'] = df_sample['filename'].values
    bow_df['title'] = df_sample['title'].values
    bow_df['label'] = df_sample['label'].values
    bow_df['total_words'] = bow_dense.sum(axis=1)
    bow_df['unique_words'] = (bow_dense > 0).sum(axis=1)
    
    bow_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f"✅ Hasil BOW disimpan di: {output_file}")
    print(f"📊 Shape file output: {bow_df.shape}")
    
    return output_file

def save_bow_model(bow_transformer, filename="bow_model.pkl"):
    """Menyimpan model BOW"""
    if not os.path.exists(CONFIG.BOW_OUTPUT_DIR):
        os.makedirs(CONFIG.BOW_OUTPUT_DIR)
    
    model_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, filename)
    
    if bow_transformer.save_bow_model(model_file):
        return model_file
    else:
        return None

def save_bow_for_lda(bow_matrix, vocabulary, filename="bow_for_lda.csv"):
    """Menyimpan BOW matrix khusus untuk LDA (tanpa metadata)"""
    if not os.path.exists(CONFIG.BOW_OUTPUT_DIR):
        os.makedirs(CONFIG.BOW_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, filename)
    
    print(f"💾 Menyimpan BOW untuk LDA...")
    
    # Hanya simpan BOW matrix tanpa metadata (format untuk LDA)
    bow_dense = bow_matrix.toarray()
    bow_df = pd.DataFrame(bow_dense, columns=vocabulary)
    
    bow_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f"✅ BOW untuk LDA disimpan di: {output_file}")
    print(f"📊 Shape: {bow_df.shape}")
    
    return output_file

def save_lda_vocabulary(vocabulary, filename="lda_vocabulary.csv"):
    """Menyimpan vocabulary untuk LDA"""
    if not os.path.exists(CONFIG.BOW_OUTPUT_DIR):
        os.makedirs(CONFIG.BOW_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, filename)
    
    vocab_df = pd.DataFrame({
        'word': vocabulary,
        'index': range(len(vocabulary))
    })
    
    vocab_df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
    print(f"✅ Vocabulary LDA disimpan di: {output_file}")
    print(f"📊 Jumlah kata: {len(vocabulary):,}")
    
    return output_file

def save_bow_summary(stats, vocab_stats, filename="bow_summary.txt"):
    """Menyimpan summary statistik BOW"""
    if not os.path.exists(CONFIG.BOW_OUTPUT_DIR):
        os.makedirs(CONFIG.BOW_OUTPUT_DIR)
    
    output_file = os.path.join(CONFIG.BOW_OUTPUT_DIR, filename)
    
    with open(output_file, 'w', encoding=CONFIG.ENCODING) as f:
        f.write("BAG-OF-WORDS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MATRIX STATISTICS:\n")
        f.write(f"Total Documents: {stats['total_documents']:,}\n")
        f.write(f"Total Features: {stats['total_features']:,}\n")
        f.write(f"Non-Zero Elements: {stats['total_non_zero_elements']:,}\n")
        f.write(f"Sparsity: {stats['sparsity']:.2%}\n")
        f.write(f"Average Words per Document: {stats['average_words_per_doc']:.1f}\n")
        f.write(f"Memory Usage: {stats['memory_size_mb']:.1f} MB\n\n")
        
        f.write("VOCABULARY STATISTICS:\n")
        f.write(f"Mean Word Frequency: {vocab_stats['word_frequency_range']['mean']:.1f}\n")
        f.write(f"Median Word Frequency: {vocab_stats['word_frequency_range']['median']:.1f}\n")
        f.write(f"Word Frequency Range: {vocab_stats['word_frequency_range']['min']} - {vocab_stats['word_frequency_range']['max']}\n\n")
        
        f.write("TOP 10 MOST FREQUENT WORDS:\n")
        for i, (word, freq) in enumerate(vocab_stats['most_frequent_words'], 1):
            f.write(f"{i:2d}. {word:<20} : {freq:>8,}\n")
    
    print(f"✅ Summary BOW disimpan di: {output_file}")
    return output_file

def test_bow():
    """Function untuk testing BOW"""
    print("=== 🧪 TESTING BAG-OF-WORDS TRANSFORMATION ===")
    start_time = datetime.now()
    
    # Load data yang sudah diproses
    df = load_processed_data()
    if df is None:
        return None, None, None
    
    # Ambil tokens_list untuk BOW
    documents = df['tokens_list'].tolist()
    labels = df['label'].tolist()
    
    print(f"\n📋 DATA OVERVIEW:")
    print(f"   • Jumlah dokumen : {len(documents):,}")
    print(f"   • Jumlah kategori: {len(pd.Series(labels).unique())}")
    
    # Tampilkan distribusi kategori
    label_counts = pd.Series(labels).value_counts()
    print(f"   • Distribusi kategori:")
    for label, count in label_counts.items():
        percentage = (count / len(labels)) * 100
        print(f"      - {label}: {count:>4} dokumen ({percentage:5.1f}%)")
    
    # Buat BOW matrix
    bow_transformer = BOWTransformer()
    bow_matrix, vocabulary = bow_transformer.create_bow_matrix(documents)
    
    # Dapatkan statistik
    stats = bow_transformer.get_matrix_stats(bow_matrix)
    vocab_stats = bow_transformer.get_vocabulary_stats(bow_matrix)
    
    # Tampilkan hasil
    display_bow_results(bow_matrix, vocabulary, stats, vocab_stats, top_n=30)
    
    # Simpan semua hasil
    print(f"\n💾 MENYIMPAN HASIL...")
    
    # 1. Simpan hasil untuk analisis (dengan metadata)
    bow_output_file = save_bow_results(bow_matrix, vocabulary, df)
    
    # 2. Simpan khusus untuk LDA (tanpa metadata)
    lda_bow_file = save_bow_for_lda(bow_matrix, vocabulary)
    
    # 3. Simpan vocabulary untuk LDA
    lda_vocab_file = save_lda_vocabulary(vocabulary)
    
    # 4. Simpan model BOW
    model_file = save_bow_model(bow_transformer)
    
    # 5. Simpan summary
    summary_file = save_bow_summary(stats, vocab_stats)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n🎉 BAG-OF-WORDS TRANSFORMATION SELESAI!")
    print(f"⏱️  Total waktu: {total_time:.2f} detik")
    print(f"\n📁 FILE YANG DIHASILKAN:")
    print(f"   • Hasil analisis BOW : {bow_output_file}")
    print(f"   • Data untuk LDA     : {lda_bow_file}")
    print(f"   • Vocabulary LDA     : {lda_vocab_file}")
    print(f"   • Model BOW          : {model_file}")
    print(f"   • Summary statistik  : {summary_file}")
    print(f"   • Total dokumen      : {stats['total_documents']:,}")
    print(f"   • Total vocabulary   : {stats['total_features']:,}")
    
    return bow_matrix, vocabulary, df

if __name__ == "__main__":
    bow_matrix, vocabulary, df = test_bow()
    
    if bow_matrix is not None:
        print(f"\n✅ Proses BOW berhasil!")
        print(f"🎯 Selanjutnya jalankan: CNP_METODE_LDA.py")
    else:
        print(f"\n❌ Proses BOW gagal!")