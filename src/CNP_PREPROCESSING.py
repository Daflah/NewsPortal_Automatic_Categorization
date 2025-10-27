'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 10 Oktober 2025
Purpose : Untuk membersihkan teks sebelum masuk ke metode bag-of-words
=============================================================================================================================
'''

import re
import os
import pandas as pd
import string
import ast
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import CONFIG

def load_stopwords():
    """Load stopwords dari file txt"""
    if not os.path.exists(CONFIG.STOPWORDS_FILE):
        print(f"Warning: Stopwords file {CONFIG.STOPWORDS_FILE} tidak ditemukan")
        return set()
    
    try:
        with open(CONFIG.STOPWORDS_FILE, 'r', encoding=CONFIG.ENCODING) as file:
            stopwords = {line.strip() for line in file if line.strip() and not line.startswith('#')}
        print(f"Stopwords loaded: {len(stopwords)} words")
        return stopwords
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()

class TextPreprocessor:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords = load_stopwords()
    
    def clean_text(self, text):
        """Membersihkan teks dari karakter tidak penting"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags, numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """Pipeline preprocessing lengkap"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
            
        tokens = cleaned.split()
        
        if not tokens:
            return []
        
        # Stemming
        try:
            stemmed = self.stemmer.stem(' '.join(tokens)).split()
        except Exception as e:
            print(f"Stemming error: {e}")
            stemmed = tokens
        
        # Stopword removal and filter short tokens
        filtered_tokens = [
            token for token in stemmed 
            if token not in self.stopwords 
            and len(token) > 2
            and token.isalpha()  # Hanya kata yang terdiri dari huruf
        ]
        
        return filtered_tokens
    
    def preprocess_documents(self, documents):
        """Preprocess multiple documents"""
        print("Memproses dokumen...")
        processed_docs = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            if i % 1000 == 0 and i > 0:
                print(f"  Diproses: {i}/{total_docs} dokumen")
            processed_docs.append(self.preprocess_text(doc))
        
        print(f"  Selesai: {total_docs} dokumen diproses")
        return processed_docs

def extract_category_from_filename(filename):
    """Extract category from filename - sekarang lebih sederhana karena sudah konsisten"""
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    # Kategori selalu di bagian kedua (index 1)
    if len(parts) >= 2:
        category = parts[1].lower()
        return category
    
    return 'unknown'

def normalize_label(category_from_file):
    """Normalisasi label untuk konsistensi - sekarang lebih sederhana"""
    label_mapping = {
        'edukasi': 'Edukasi',
        'finance': 'Finance', 
        'health': 'Health',
        'lifestyle': 'Lifestyle',
        'nasional': 'Nasional',
        'otomotif': 'Otomotif',
        'sport': 'Sport',
        'tekno': 'Teknologi',
        'teknologi': 'Teknologi',
        'travel': 'Travel'
    }
    
    normalized = label_mapping.get(category_from_file.lower(), category_from_file.title())
    return normalized

def load_corpus_files():
    """Load file CSV dari corpus"""
    if not os.path.exists(CONFIG.DATA_PATH):
        print(f"Error: Path {CONFIG.DATA_PATH} tidak ditemukan")
        return [], [], [], []
    
    csv_files = [CONFIG.TEST_FILE] if CONFIG.TESTING_MODE else \
                [f for f in os.listdir(CONFIG.DATA_PATH) if f.endswith('.csv')]
    
    print(f"=== MODE {'TESTING' if CONFIG.TESTING_MODE else 'PRODUCTION'} ===")
    print(f"Menemukan {len(csv_files)} file CSV di folder corpus")
    
    documents, labels, titles, filenames = [], [], [], []
    file_stats = {}
    total_documents = 0
    
    for filename in csv_files:
        filepath = os.path.join(CONFIG.DATA_PATH, filename)
        try:
            # Baca CSV
            df = pd.read_csv(filepath, sep=CONFIG.CSV_SEPARATOR, encoding=CONFIG.ENCODING)
            
            if CONFIG.TESTING_MODE:
                df = df.head(CONFIG.TEST_SAMPLE_SIZE)
            
            # Check required columns
            if CONFIG.TEXT_COLUMN not in df.columns:
                print(f"Warning: Kolom '{CONFIG.TEXT_COLUMN}' tidak ditemukan di {filename}")
                continue
            
            # Extract category from filename
            category_from_file = extract_category_from_filename(filename)
            normalized_label = normalize_label(category_from_file)
            
            # Extract data
            texts = df[CONFIG.TEXT_COLUMN].fillna('').astype(str).tolist()
            valid_texts = [text for text in texts if text.strip()]
            
            if not valid_texts:
                print(f"Warning: Tidak ada teks valid di {filename}")
                continue
            
            documents.extend(valid_texts)
            filenames.extend([filename] * len(valid_texts))
            
            # Extract titles
            if CONFIG.TITLE_COLUMN in df.columns:
                titles.extend(df[CONFIG.TITLE_COLUMN].fillna('').astype(str).tolist())
            else:
                titles.extend([''] * len(valid_texts))
            
            # Use normalized label from filename for all documents
            labels.extend([normalized_label] * len(valid_texts))
            
            file_stats[filename] = len(valid_texts)
            total_documents += len(valid_texts)
            
            print(f"  ✅ {filename}: {len(valid_texts)} dokumen → Kategori: {normalized_label}")
            
        except Exception as e:
            print(f"❌ Error membaca {filename}: {e}")
            continue
    
    print(f"\n📊 Total berhasil load {total_documents} dokumen dari {len(file_stats)} file")
    
    # Print summary by category
    if labels:
        print(f"\n📈 Ringkasan per Kategori:")
        label_series = pd.Series(labels)
        category_summary = label_series.value_counts().sort_index()
        for category, count in category_summary.items():
            percentage = (count / total_documents) * 100
            print(f"  - {category}: {count} dokumen ({percentage:.1f}%)")
    
    return documents, labels, titles, filenames

def get_dataframe_with_processed_text():
    """Return DataFrame dengan teks yang sudah diproses"""
    print("\n🎯 Memuat dan memproses data...")
    documents, labels, titles, filenames = load_corpus_files()
    
    if not documents:
        print("❌ Tidak ada data yang berhasil dimuat")
        return pd.DataFrame()
    
    print("\n🔧 Memulai preprocessing teks...")
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'title': titles,
        'original_text': documents,
        'processed_text': [' '.join(tokens) for tokens in processed_docs],
        'tokens': [str(tokens) for tokens in processed_docs],
        'tokens_list': processed_docs,
        'label': labels
    })
    
    # Filter out empty processed documents
    initial_count = len(df)
    df = df[df['processed_text'].str.len() > 0]
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"⚠️  {filtered_count} dokumen kosong dihapus setelah preprocessing")
    
    # Statistics
    original_words = sum(len(str(text).split()) for text in documents)
    processed_words = sum(len(tokens) for tokens in processed_docs)
    
    print(f"\n=== 📊 STATISTIK PREPROCESSING ===")
    print(f"Jumlah dokumen: {len(df)}")
    print(f"Jumlah kata sebelum preprocessing: {original_words:,}")
    print(f"Jumlah kata setelah preprocessing: {processed_words:,}")
    
    if original_words > 0:
        reduction = ((original_words - processed_words) / original_words * 100)
        print(f"Reduksi vocabulary: {reduction:.2f}%")
    
    # Token statistics
    token_lengths = [len(tokens) for tokens in processed_docs]
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    print(f"Rata-rata tokens per dokumen: {avg_tokens:.1f}")
    
    if not df.empty and 'label' in df.columns:
        print(f"\n📂 Distribusi Kategori Akhir:")
        label_counts = df['label'].value_counts().sort_index()
        for category, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {category}: {count} dokumen ({percentage:.1f}%)")
    
    return df

def save_processed_data(df):
    """Save processed data to CSV"""
    if not os.path.exists(CONFIG.PROCESSED_DATA_PATH):
        os.makedirs(CONFIG.PROCESSED_DATA_PATH)
        print(f"📁 Directory created: {CONFIG.PROCESSED_DATA_PATH}")
    
    if df.empty:
        print("❌ Tidak ada data untuk disimpan")
        return None
    
    filename = f"processed_data_test_{CONFIG.TEST_FILE.split('.')[0]}.csv" if CONFIG.TESTING_MODE else "processed_data_full.csv"
    output_file = os.path.join(CONFIG.PROCESSED_DATA_PATH, filename)
    
    try:
        df.to_csv(output_file, index=False, encoding=CONFIG.ENCODING)
        print(f"💾 Data tersimpan di: {output_file}")
        print(f"📊 Shape: {df.shape}")
        return output_file
    except Exception as e:
        print(f"❌ Error menyimpan data: {e}")
        return None

def load_tokens_from_csv(csv_file):
    """Load tokens dari CSV file dan convert kembali ke list"""
    try:
        df = pd.read_csv(csv_file, encoding=CONFIG.ENCODING)
        
        # Convert string representation back to list
        df['tokens_list'] = df['tokens'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else []
        )
        
        print(f"✅ Loaded {len(df)} documents from {csv_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading tokens: {e}")
        return pd.DataFrame()

def analyze_dataset_quality(df):
    """Analisis kualitas dataset setelah preprocessing"""
    if df.empty:
        return
    
    print(f"\n=== 🔍 ANALISIS KUALITAS DATASET ===")
    
    # Token length distribution
    token_lengths = [len(tokens) for tokens in df['tokens_list']]
    avg_tokens = sum(token_lengths) / len(token_lengths)
    max_tokens = max(token_lengths)
    min_tokens = min(token_lengths)
    
    print(f"Panjang Token:")
    print(f"  - Rata-rata: {avg_tokens:.1f} tokens/dokumen")
    print(f"  - Maksimum: {max_tokens} tokens")
    print(f"  - Minimum: {min_tokens} tokens")
    
    # Check for very short documents
    short_docs = len([length for length in token_lengths if length < 5])
    if short_docs > 0:
        print(f"⚠️  {short_docs} dokumen memiliki kurang dari 5 tokens")
    
    # Category balance
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        imbalance_ratio = label_counts.max() / label_counts.min()
        print(f"\n⚖️  Balance Dataset:")
        print(f"  - Rasio ketidakseimbangan: {imbalance_ratio:.2f}")
        if imbalance_ratio > 5:
            print("  ⚠️  Dataset sangat tidak seimbang")
        elif imbalance_ratio > 3:
            print("  ⚠️  Dataset cukup tidak seimbang")
        else:
            print("  ✅ Dataset cukup seimbang")

def test_preprocessing():
    """Function untuk testing preprocessing"""
    print("=== 🔧 TESTING PREPROCESSING ===")
    
    df = get_dataframe_with_processed_text()
    if df.empty:
        print("❌ Tidak ada data yang berhasil diproses")
        return
    
    print(f"\n📋 DataFrame shape: {df.shape}")
    print(f"📝 Kolom: {list(df.columns)}")
    
    # Show samples from different categories - UBAH: ambil 5 sample
    print(f"\n=== 🧪 CONTOH HASIL PREPROCESSING (5 SAMPLE) ===")
    
    # Ambil 5 sample acak dari dataset, pastikan mewakili berbagai kategori
    if len(df) >= 5:
        # Coba ambil dari kategori yang berbeda
        sampled_df = df.groupby('label').head(1).reset_index(drop=True)
        
        # Jika kurang dari 5 kategori, tambahkan sample acak
        if len(sampled_df) < 5:
            additional_samples = df[~df.index.isin(sampled_df.index)].head(5 - len(sampled_df))
            sampled_df = pd.concat([sampled_df, additional_samples]).reset_index(drop=True)
        else:
            sampled_df = sampled_df.head(5)
    else:
        sampled_df = df.head(min(5, len(df)))
    
    for i, row in sampled_df.iterrows():
        print(f"\n--- Sample {i+1} ({row['label']}) ---")
        print(f"📄 File: {row['filename']}")
        print(f"📰 Judul: {row['title'][:60]}..." if row['title'] and row['title'].strip() else "📰 Judul: (tidak ada)")
        
        original = row['original_text']
        processed = row['processed_text']
        tokens_list = ast.literal_eval(row['tokens'])
        
        print(f"📖 Original: {original[:80]}...")
        print(f"📊 Panjang original: {len(original)} karakter, {len(original.split())} kata")
        print(f"🔧 Processed: {processed[:80]}...")
        print(f"📊 Panjang processed: {len(processed)} karakter")
        
        # UBAH: Tampilkan 20 tokens pertama
        print(f"🔤 Tokens (20 pertama): {tokens_list[:20]}")
        print(f"📈 Jumlah tokens: {len(tokens_list)}")
        
        if len(original.split()) > 0:
            reduction = ((len(original.split()) - len(tokens_list)) / len(original.split()) * 100)
            print(f"📉 Reduksi kata: {reduction:.2f}%")
    
    # Analyze dataset quality
    analyze_dataset_quality(df)
    
    # Save data
    output_file = save_processed_data(df)
    
    if output_file:
        print(f"\n=== ✅ PREPROCESSING SELESAI ===")
        print(f"💾 File hasil: {output_file}")
        print(f"📊 Total dokumen: {len(df)}")
        
        # Test loading back
        print(f"\n=== 🔄 TEST LOADING BACK ===")
        loaded_df = load_tokens_from_csv(output_file)
        if not loaded_df.empty:
            print(f"✅ Berhasil load kembali {len(loaded_df)} dokumen")
            
            # Verify categories
            unique_categories = loaded_df['label'].unique()
            print(f"🎯 Kategori unik: {sorted(unique_categories)}")
            
            # Tampilkan 5 sample dari data yang diload kembali
            print(f"\n=== 🧪 VERIFIKASI DATA YANG DILOAD KEMBALI (5 SAMPLE) ===")
            verification_samples = loaded_df.head(5)
            
            for i, row in verification_samples.iterrows():
                print(f"\n--- Verification Sample {i+1} ({row['label']}) ---")
                print(f"📄 File: {row['filename']}")
                tokens_list = row['tokens_list']
                print(f"🔤 Tokens (20 pertama): {tokens_list[:20]}")
                print(f"📈 Jumlah tokens: {len(tokens_list)}")
                
    else:
        print(f"\n=== ❌ PREPROCESSING GAGAL ===")

if __name__ == "__main__":
    test_preprocessing()