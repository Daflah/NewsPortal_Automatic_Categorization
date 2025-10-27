'''
=============================================================================================================================
Author  : Daflah Tsany Gusra
Build   : 10 Oktober 2025
Purpose : Setting global untuk penyesuaian tanpa mengubah pada coding langsung
=============================================================================================================================
'''

from pathlib import Path

# ==================== PATH SETTINGS ====================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "corpus/"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed/" 
OUTPUT_PATH = BASE_DIR / "output/"
STOPWORDS_FILE = BASE_DIR / "data" / "stopwordremoval" / "stopword_list.txt"

# ==================== TESTING MODE PREPROCESS ====================
TESTING_MODE = False
TEST_FILE = "detik_finance_20251019_025856.csv"
TEST_SAMPLE_SIZE = 5

# ==================== CSV SETTINGS ====================
CSV_SEPARATOR = ","
ENCODING = "utf-8"
TEXT_COLUMN = "Content"
LABEL_COLUMN = "Category"
TITLE_COLUMN = "Title"
LINK_COLUMN = "Link"
TIME_COLUMN = "Time"

# ==================== PREPROCESSING SETTINGS ====================
STEMMER_TYPE = "sastrawi"
LANGUAGE = "indonesian"
INDONESIAN_STOPWORDS = []  

# ==================== BOW SETTINGS ====================
MAX_FEATURES = 12000
MIN_DF = 2
MAX_DF = 0.85
BOW_OUTPUT_DIR = OUTPUT_PATH / "BoW" 

# ==================== LDA SETTINGS ====================
N_TOPICS = 9
LDA_ITERATIONS = 800
LDA_OUTPUT_DIR = OUTPUT_PATH / "LDA"

# ==================== RANDOM FOREST SETTINGS ====================
RF_ESTIMATORS = 100
RF_RANDOM_STATE = 42
MODEL_RESULTS_DIR = OUTPUT_PATH / "model_results_save" 

# ==================== CATEGORY MAPPING ====================
CATEGORY_MAPPING = {
    'teknologi': 0,
    'otomotif': 1, 
    'sport': 2,
    'lifestyle': 3,
    'finance': 4,
    'health': 5,
    'education': 6,
    'nasional': 7,
    'travel': 8
}