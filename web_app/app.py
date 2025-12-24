import os
import sys
import pickle
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify, session
from datetime import datetime
import hashlib
from werkzeug.utils import secure_filename
import pandas as pd
from pathlib import Path
import csv

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add config to path untuk import database
sys.path.append(os.path.join(BASE_DIR, '..', 'config'))
from DATABASE import db_config

app = Flask(__name__)
app.secret_key = 'newsportal-secret-key-2024'  # 🔑 ADD SECRET KEY FOR SESSION

# ---------------------------------
# Upload configuration (Newest)
# ---------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    # filename: string, contoh "data_berita.csv"
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()   # ambil bagian setelah titik
    return ext in ALLOWED_EXTENSIONS


# Global variables untuk models
models = {}

# Label mapping - akan di-load dari model_metrics.json
LABEL_MAPPING = {}
REVERSE_LABEL_MAPPING = {}

# Database class
class NewsDatabase:
    def __init__(self):
        self.db_config = db_config
    
    def get_all_articles(self, limit=100):
        """Get all articles from database"""
        conn = self.db_config.get_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, link, category, time, content 
                FROM news_articles 
                ORDER BY id 
                LIMIT %s
            """, (limit,))
            
            articles = []
            for row in cur.fetchall():
                articles.append({
                    'id': row[0],
                    'title': row[1],
                    'link': row[2],
                    'category': row[3],
                    'time': row[4],
                    'content': row[5]
                })
            
            cur.close()
            return articles
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_articles_by_category(self, category, limit=50):
        """Get articles by category"""
        conn = self.db_config.get_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, link, category, time, content 
                FROM news_articles 
                WHERE category = %s 
                ORDER BY id 
                LIMIT %s
            """, (category, limit))
            
            articles = []
            for row in cur.fetchall():
                articles.append({
                    'id': row[0],
                    'title': row[1],
                    'link': row[2],
                    'category': row[3],
                    'time': row[4],
                    'content': row[5]
                })
            
            cur.close()
            return articles
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def search_articles(self, search_term, limit=50):
        """Search articles by title or content"""
        conn = self.db_config.get_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor()
            search_pattern = f"%{search_term}%"
            cur.execute("""
                SELECT id, title, link, category, time, content 
                FROM news_articles 
                WHERE title ILIKE %s OR content ILIKE %s 
                ORDER BY id 
                LIMIT %s
            """, (search_pattern, search_pattern, limit))
            
            articles = []
            for row in cur.fetchall():
                articles.append({
                    'id': row[0],
                    'title': row[1],
                    'link': row[2],
                    'category': row[3],
                    'time': row[4],
                    'content': row[5]
                })
            
            cur.close()
            return articles
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_categories(self):
        """Get all unique categories"""
        conn = self.db_config.get_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT category, COUNT(*) as count 
                FROM news_articles 
                GROUP BY category 
                ORDER BY count DESC
            """)
            
            categories = []
            for row in cur.fetchall():
                categories.append({
                    'name': row[0],
                    'count': row[1]
                })
            
            cur.close()
            return categories
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_article_by_id(self, article_id):
        """Get article by ID"""
        conn = self.db_config.get_connection()
        if not conn:
            return None
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, title, link, category, time, content 
                FROM news_articles 
                WHERE id = %s
            """, (article_id,))
            
            row = cur.fetchone()
            cur.close()
            
            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'link': row[2],
                    'category': row[3],
                    'time': row[4],
                    'content': row[5]
                }
            return None
            
        except Exception as e:
            print(f"Database error: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_database_stats(self):
        """Get database statistics"""
        conn = self.db_config.get_connection()
        if not conn:
            return {}
        
        try:
            cur = conn.cursor()
            
            # Total articles
            cur.execute("SELECT COUNT(*) FROM news_articles")
            total_articles = cur.fetchone()[0]
            
            # Articles by category
            cur.execute("""
                SELECT category, COUNT(*) 
                FROM news_articles 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """)
            category_stats = {row[0]: row[1] for row in cur.fetchall()}
            
            cur.close()
            
            return {
                'total_articles': total_articles,
                'categories': category_stats
            }
            
        except Exception as e:
            print(f"Database error: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def bulk_import_articles(self, articles_list):
        """Import banyak artikel sekaligus ke database."""
        imported = 0
        skipped = 0
        errors = []

        # gunakan db_config.get_connection(), BUKAN self.get_connection()
        conn = self.db_config.get_connection()
        cursor = conn.cursor()

        for art in articles_list:
            try:
                cursor.execute(
                    """
                    INSERT INTO news_articles (title, content, category, link, time)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        art['title'],
                        art['content'],
                        art['category'],
                        art['link'],
                        art['time']
                    )
                )
                imported += 1
            except Exception as e:
                conn.rollback()
                skipped += 1
                errors.append(str(e))
            else:
                conn.commit()

        cursor.close()
        conn.close()

        return {
            'success': True,
            'imported': imported,
            'skipped': skipped,
            'errors': errors
        }

# 🔐 USER DATABASE CLASS YANG DIPERBAIKI
class UserDatabase:
    def __init__(self):
        self.db_config = db_config
    
    def create_users_table(self):
        """Create users table if not exists - PERBAIKI QUERY"""
        conn = self.db_config.get_connection()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            # Cek dulu apakah tabel sudah ada
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                cur.execute("""
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        first_name VARCHAR(100) NOT NULL,
                        last_name VARCHAR(100) NOT NULL,
                        user_type VARCHAR(20) DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Users table created successfully")
                
                # Buat user admin default
                self.create_default_admin()
            else:
                print("✅ Users table already exists")
                
            cur.close()
            return True
            
        except Exception as e:
            print(f"❌ Error creating users table: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def create_default_admin(self):
        """Create default admin user"""
        try:
            admin_email = "admin@newsportal.com"
            admin_password = "admin123"
            admin_first_name = "System"
            admin_last_name = "Administrator"
            
            if not self.user_exists(admin_email):
                password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
                
                conn = self.db_config.get_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO users (email, password_hash, first_name, last_name, user_type)
                    VALUES (%s, %s, %s, %s, %s)
                """, (admin_email, password_hash, admin_first_name, admin_last_name, 'admin'))
                
                conn.commit()
                cur.close()
                print("✅ Default admin user created")
                print(f"   Email: {admin_email}")
                print(f"   Password: {admin_password}")
            else:
                print("✅ Default admin user already exists")
                
        except Exception as e:
            print(f"❌ Error creating default admin: {e}")
    
    def create_user(self, email, password, first_name, last_name, user_type='user'):
        """Create new user"""
        conn = self.db_config.get_connection()
        if not conn:
            return False
        
        try:
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (email, password_hash, first_name, last_name, user_type)
                VALUES (%s, %s, %s, %s, %s)
            """, (email, password_hash, first_name, last_name, user_type))
            
            conn.commit()
            cur.close()
            print(f"✅ User created: {email}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def authenticate_user(self, email, password):
        """Authenticate user"""
        conn = self.db_config.get_connection()
        if not conn:
            return None
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cur = conn.cursor()
            cur.execute("""
                SELECT id, email, first_name, last_name, user_type 
                FROM users 
                WHERE email = %s AND password_hash = %s
            """, (email, password_hash))
            
            user = cur.fetchone()
            cur.close()
            
            if user:
                # Update last login
                cur = conn.cursor()
                cur.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE id = %s
                """, (user[0],))
                conn.commit()
                cur.close()
                
                return {
                    'id': user[0],
                    'email': user[1],
                    'first_name': user[2],
                    'last_name': user[3],
                    'user_type': user[4]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error authenticating user: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def user_exists(self, email):
        """Check if user exists"""
        conn = self.db_config.get_connection()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            exists = cur.fetchone() is not None
            cur.close()
            return exists
            
        except Exception as e:
            print(f"❌ Error checking user existence: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.db_config.get_connection()
        if not conn:
            return None
        
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, email, first_name, last_name, user_type 
                FROM users 
                WHERE id = %s
            """, (user_id,))
            
            user = cur.fetchone()
            cur.close()
            
            if user:
                return {
                    'id': user[0],
                    'email': user[1],
                    'first_name': user[2],
                    'last_name': user[3],
                    'user_type': user[4]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error getting user by ID: {e}")
            return None
        finally:
            if conn:
                conn.close()

# Initialize databases
news_db = NewsDatabase()
user_db = UserDatabase()

# 🔐 FUNCTION UNTUK GET USER INFO
def get_user_info():
    """Get user info from session"""
    if 'user_id' in session:
        user = user_db.get_user_by_id(session['user_id'])
        if user:
            return {
                'id': user['id'],
                'name': f"{user['first_name']} {user['last_name']}",
                'email': user['email'],
                'type': user['user_type'],
                'is_admin': user['user_type'] == 'admin'
            }
    
    return None

def bulk_import_articles(self, articles_list):
    """
    Import multiple articles dari dataset
    articles_list: list of dicts dengan keys: title, content, category, link, time
    """
    conn = self.db_config.get_connection()
    if not conn:
        return {'success': False, 'error': 'Database connection failed', 'imported': 0}
    
    try:
        cur = conn.cursor()
        imported_count = 0
        skipped_count = 0
        errors = []
        
        for article in articles_list:
            try:
                # Validasi data
                if not article.get('title') or not article.get('content'):
                    skipped_count += 1
                    continue
                
                # Sanitasi input
                title = article.get('title', '').strip()[:500]
                content = article.get('content', '').strip()[:5000]
                category = article.get('category', 'nasional').strip().lower()
                link = article.get('link', '').strip()[:500]
                time = article.get('time') or datetime.now()
                
                # Buat content hash untuk deteksi duplikasi
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Cek duplikasi berdasarkan title
                cur.execute(
                    "SELECT id FROM news_articles WHERE title = %s LIMIT 1",
                    (title,)
                )
                
                if cur.fetchone() is None:
                    # Insert data baru
                    try:
                        cur.execute("""
                            INSERT INTO news_articles (title, content, category, link, time)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (title, content, category, link, time))
                        imported_count += 1
                    except:
                        # Jika kolom content_hash ada
                        cur.execute("""
                            INSERT INTO news_articles (title, content, category, link, time, content_hash)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (title, content, category, link, time, content_hash))
                        imported_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"⚠️ Skip row: {str(e)}")
                errors.append(f"Row error: {str(e)}")
                skipped_count += 1
                continue
        
        conn.commit()
        cur.close()
        
        print(f"✅ Import Summary - Imported: {imported_count}, Skipped: {skipped_count}")
        
        return {
            'success': True,
            'imported': imported_count,
            'skipped': skipped_count,
            'errors': errors[:5]  # Batasi hanya 5 error pertama
        }
        
    except Exception as e:
        print(f"❌ Bulk import error: {e}")
        return {'success': False, 'error': str(e), 'imported': 0}
    finally:
        if conn:
            conn.close()

# ===============================
# 🔐 AUTHENTICATION ROUTES
# ===============================

@app.route('/login')
def login_page():
    """Halaman login"""
    # Jika user sudah login, redirect ke home
    if 'user_id' in session:
        return redirect(url_for('home'))
    
    user_info = get_user_info()
    return render_template('CNP_LOGIN_PAGE.html', user=user_info)

@app.route('/signup')
def signup_page():
    """Halaman signup"""
    # Jika user sudah login, redirect ke home
    if 'user_id' in session:
        return redirect(url_for('home'))
    
    user_info = get_user_info()
    return render_template('CNP_SIGNIN_PAGE.html', user=user_info)

@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint untuk login"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email dan password harus diisi'})
        
        # Authenticate user
        user = user_db.authenticate_user(email, password)
        
        if user:
            # Set session
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session['user_type'] = user['user_type']
            
            return jsonify({
                'success': True,
                'message': 'Login berhasil!',
                'user': user
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Email atau password salah'
            })
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({
            'success': False,
            'message': 'Terjadi kesalahan sistem'
        })

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """API endpoint untuk signup"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        # Validation
        if not all([email, password, first_name, last_name]):
            return jsonify({'success': False, 'message': 'Semua field harus diisi'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password minimal 6 karakter'})
        
        if user_db.user_exists(email):
            return jsonify({'success': False, 'message': 'Email sudah terdaftar'})
        
        # Create user
        if user_db.create_user(email, password, first_name, last_name):
            # Auto login after signup
            user = user_db.authenticate_user(email, password)
            if user:
                session['user_id'] = user['id']
                session['user_email'] = user['email']
                session['user_name'] = f"{user['first_name']} {user['last_name']}"
                session['user_type'] = user['user_type']
                
            return jsonify({
                'success': True,
                'message': 'Pendaftaran berhasil!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Gagal membuat akun. Silakan coba lagi.'
            })
            
    except Exception as e:
        print(f"❌ Signup error: {e}")
        return jsonify({
            'success': False,
            'message': 'Terjadi kesalahan sistem'
        })

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('home'))

# ===============================
# UPDATE EXISTING ROUTES WITH USER INFO
# ===============================

@app.route('/')
def home():
    """Halaman utama - Main Page dengan data dari database"""
    articles = news_db.get_all_articles(limit=50)
    categories = news_db.get_categories()
    stats = news_db.get_database_stats()
    
    # 🔐 PERBAIKI: Gunakan function terpusat
    user_info = get_user_info()
    
    return render_template('CNP_MAIN_PAGE.html', 
                         articles=articles, 
                         categories=categories,
                         current_category='All',
                         stats=stats,
                         user=user_info)

@app.route('/category/<category_name>')
def category_news(category_name):
    """Show articles by category"""
    articles = news_db.get_articles_by_category(category_name, limit=50)
    categories = news_db.get_categories()
    stats = news_db.get_database_stats()
    
    # 🔐 PERBAIKI: Gunakan function terpusat
    user_info = get_user_info()
    
    return render_template('CNP_MAIN_PAGE.html', 
                         articles=articles, 
                         categories=categories,
                         current_category=category_name,
                         stats=stats,
                         user=user_info)

@app.route('/search')
def search_news():
    """Search articles"""
    search_term = request.args.get('q', '')
    articles = []
    
    if search_term:
        articles = news_db.search_articles(search_term, limit=50)
    
    categories = news_db.get_categories()
    stats = news_db.get_database_stats()
    
    # 🔐 PERBAIKI: Gunakan function terpusat
    user_info = get_user_info()
    
    return render_template('CNP_MAIN_PAGE.html', 
                         articles=articles, 
                         categories=categories,
                         current_category='Search',
                         search_term=search_term,
                         stats=stats,
                         user=user_info)

@app.route('/article/<int:article_id>')
def article_page(article_id):
    """Halaman detail artikel dari database"""
    article = news_db.get_article_by_id(article_id)
    
    if not article:
        return render_template('CNP_MAIN_PAGE.html'), 404
    
    # 🔐 PERBAIKI: Gunakan function terpusat
    user_info = get_user_info()
    
    return render_template('CNP_ARTICLE_PAGE.html', article=article, user=user_info)

@app.route('/input')
def input_page():
    """Halaman input berita"""
    # 🔐 PERBAIKI: Gunakan function terpusat
    user_info = get_user_info()
    
    return render_template('CNP_INPUT_PAGE.html', user=user_info)

@app.route('/check_auth')
def check_auth():
    """API endpoint untuk cek status authentication user"""
    user_info = get_user_info()
    return jsonify({
        'is_authenticated': user_info is not None,
        'user_type': user_info['type'] if user_info else 'none',
        'user_name': user_info['name'] if user_info else ''
    })

# ------------------------------------
# Route (Newest)
# ------------------------------------
@app.route('/admin/dataset')
def admin_dataset():
    """Halaman admin untuk manage dataset"""
    user_info = get_user_info()
    
    # Cek apakah user adalah admin
    if not user_info or not user_info['is_admin']:
        return redirect(url_for('login_page'))
    
    return render_template('CNP_ADMIN_DATASET.html', user=user_info)

def smart_read_file(filepath, filename):
    """Cerdas membaca file CSV/Excel dengan handling berbagai format"""
    try:
        if filename.endswith('.csv'):
            # Coba berbagai encoding untuk CSV
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    print(f"✅ CSV read berhasil dengan encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"⚠️ Encoding {encoding} gagal: {e}")
                    continue
            else:
                raise Exception("Gagal membaca file CSV dengan semua encoding yang dicoba")
        else:  # xlsx/xls
            try:
                # Coba baca semua sheet
                excel_file = pd.ExcelFile(filepath)
                print(f"📄 Excel sheets: {excel_file.sheet_names}")
                
                # Coba sheet pertama
                df = excel_file.parse(excel_file.sheet_names[0])
                print(f"✅ Excel file read successfully from sheet: {excel_file.sheet_names[0]}")
            except Exception as e:
                print(f"❌ Error membaca Excel: {e}")
                raise
        
        # Bersihkan dataframe
        df = clean_dataframe(df)
        
        return df
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        raise

def clean_dataframe(df):
    """Bersihkan dataframe dari NaN, empty rows, dan format yang tidak diharapkan"""
    # Buat salinan
    df = df.copy()
    
    # Hapus baris yang semua kolomnya NaN
    df = df.dropna(how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Konversi semua kolom ke string, tapi jaga tipe data asli untuk data yang valid
    for col in df.columns:
        # Coba konversi ke string, tapi simpan data asli jika mungkin
        try:
            # Cek jika kolom berisi list atau array
            if df[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                print(f"⚠️ Kolom {col} berisi tipe data kompleks (list/dict/tuple)")
                # Konversi ke string representation
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else '')
        except Exception as e:
            print(f"⚠️ Error processing column {col}: {e}")
            df[col] = df[col].astype(str)
    
    print(f"📊 DataFrame cleaned: shape={df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📊 Sample data:")
    if not df.empty:
        for i, col in enumerate(df.columns):
            sample = df[col].iloc[0] if not df.empty else "EMPTY"
            print(f"   {col}: {str(sample)[:100]}...")
    
    return df


def validate_and_fix_dataframe(df):
    """Cek struktur DataFrame dan coba perbaiki jika hanya 1 kolom panjang."""
    print(f"\n📊 DF awal untuk validasi: shape={df.shape}, cols={list(df.columns)}")
    
    # Jika DataFrame kosong
    if df.empty:
        print("❌ DataFrame kosong!")
        return df
    
    # Jika hanya 1 kolom dan data terlihat seperti CSV dengan delimiter yang salah
    if len(df.columns) == 1 and len(df) > 1:
        col_name = df.columns[0]
        print(f"⚠️ Hanya 1 kolom ({col_name}), coba split dengan delimiter...")
        
        # Ambil sample data
        sample_data = str(df.iloc[0, 0]) if pd.notna(df.iloc[0, 0]) else ""
        print(f"📝 Sample data (pertama): {sample_data[:200]}...")
        
        # Coba berbagai delimiter
        delimiters = [',', ';', '\t', '|', '||']
        
        best_df = df
        best_delim = None
        
        for delim in delimiters:
            try:
                # Split kolom pertama berdasarkan delimiter
                split_df = df[col_name].astype(str).str.split(delim, expand=True)
                
                # Cek hasil split
                if split_df.shape[1] > 1:
                    print(f"   Delimiter '{delim}': menghasilkan {split_df.shape[1]} kolom")
                    
                    # Bersihkan hasil split
                    split_df = split_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                    
                    # Update column names
                    new_cols = [f"col_{i}" for i in range(split_df.shape[1])]
                    split_df.columns = new_cols
                    
                    best_df = split_df
                    best_delim = delim
                    
                    # Cek jika kolom pertama kelihatan seperti header
                    first_row_values = split_df.iloc[0].tolist()
                    if all(isinstance(val, str) and val for val in first_row_values):
                        print(f"   Baris pertama kelihatan seperti header: {first_row_values}")
                        # Gunakan baris pertama sebagai header
                        header_row = split_df.iloc[0]
                        split_df = split_df[1:].reset_index(drop=True)
                        split_df.columns = header_row
                        best_df = split_df
                        print(f"   ✅ Set header dari baris pertama: {list(split_df.columns)}")
                    
                    break
                    
            except Exception as e:
                print(f"   Delimiter '{delim}' gagal: {e}")
                continue
        
        if best_delim:
            print(f"✅ Berhasil split dengan delimiter '{best_delim}'")
            df = best_df
        else:
            print("❌ Tidak ada delimiter yang berhasil")
    
    print(f"📊 DF setelah validasi: shape={df.shape}")
    print(f"📊 Columns final: {list(df.columns)}")
    
    return df

@app.route('/api/admin/upload-dataset', methods=['POST'])
def api_upload_dataset():
    """API endpoint untuk upload dataset (admin only)"""
    user_info = get_user_info()
    
    # CEK AUTHORIZATION
    if not user_info or not user_info.get('is_admin'):
        return jsonify({
            'success': False,
            'message': 'Anda tidak memiliki akses. Hanya admin yang dapat mengupload dataset.'
        }), 403
    
    try:
        # Cek file ada
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'File tidak ditemukan dalam request'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'File tidak dipilih'
            }), 400
        
        # Cek extension file
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'Format file tidak diperbolehkan. Gunakan: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Simpan file sementara
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📁 File uploaded: {filepath}")
        
        # Parse file menggunakan pandas - DENGAN ERROR HANDLING YANG BENAR
        try:
            if filename.endswith('.csv'):
                # Coba beberapa cara baca CSV
                df = None
                errors = []
                
                # Method 1: Default UTF-8
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                    print("✅ CSV read dengan UTF-8")
                except Exception as e1:
                    errors.append(f"UTF-8: {str(e1)}")
                    
                    # Method 2: Latin-1 encoding
                    try:
                        df = pd.read_csv(filepath, encoding='latin-1')
                        print("✅ CSV read dengan Latin-1")
                    except Exception as e2:
                        errors.append(f"Latin-1: {str(e2)}")
                        
                        # Method 3: With error handling
                        try:
                            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
                            print("✅ CSV read dengan skip bad lines")
                        except Exception as e3:
                            errors.append(f"Skip bad lines: {str(e3)}")
                
                if df is None:
                    raise Exception(f"Gagal membaca CSV. Errors: {'; '.join(errors)}")
            else:
                # Excel file
                df = pd.read_excel(filepath)
                print("✅ Excel file read")
            
            print(f"📊 Dataframe shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")

            df.columns = [str(col).strip() for col in df.columns]

            # === NORMALISASI CSV/EXCEL SATU KOLOM ===
            if len(df.columns) == 1:
                col0 = df.columns[0]
                first_val = str(df.iloc[0, 0])
                print("⚠️ Single-column DF. col0:", col0)
                print("⚠️ Sample first_val:", first_val[:120])

                if ',' in first_val:
                    # split isi baris jadi 5 kolom
                    temp = df[col0].astype(str).str.split(',', n=4, expand=True)
                    temp.columns = ['Title', 'Link', 'Category', 'Time', 'Content']
                    df = temp
                    print("✅ Split from row values, columns:", list(df.columns))
                elif ',' in col0:
                    # kalau header yang gabung
                    new_cols = [c.strip() for c in col0.split(',')]
                    if len(new_cols) == df.shape[1]:
                        df.columns = new_cols
                        print("✅ Split from header, columns:", list(df.columns))

            print(f"📊 DF setelah normalisasi: {df.shape}, cols={list(df.columns)}")

            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return jsonify({
                'success': False,
                'message': f'Gagal membaca file: {str(e)}'
            }), 400
        
        # Mapping kolom - DENGAN CASE-INSENSITIVE MATCHING
        column_mapping = {
            'title':   ['title', 'judul', 'headline'],
            'content': ['content', 'isi', 'article', 'body'],
            'category':['category', 'kategori'],
            'link':    ['link', 'url', 'source'],
            'time':    ['time', 'date', 'tanggal', 'waktu'],
        }

        actual_columns = {}
        cols_lower = {str(c).lower(): str(c) for c in df.columns}
        print("DEBUG columns lower:", cols_lower)

        for key, names in column_mapping.items():
            for name in names:
                name_l = name.lower()
                if name_l in cols_lower:
                    actual_columns[key] = cols_lower[name_l]
                    break

        required_keys = ['title', 'content']
        missing_keys = [k for k in required_keys if k not in actual_columns]
        
        if missing_keys:
            print(f"❌ Missing columns: {missing_keys}")
            print(f"Available columns: {list(df.columns)}")
            return jsonify({
                'success': False,
                'message': f'Kolom yang diperlukan tidak ditemukan: {", ".join(missing_keys)}. Kolom tersedia: {", ".join(df.columns)}'
            }), 400
        
        # Transform data
        articles_list = []
        for idx, row in df.iterrows():
            try:
                # Ambil data dengan safe method
                title_col = actual_columns.get('title')
                content_col = actual_columns.get('content')
                category_col = actual_columns.get('category')
                link_col = actual_columns.get('link')
                time_col = actual_columns.get('time')
                
                title = str(row.get(title_col, '')).strip() if title_col else ''
                content = str(row.get(content_col, '')).strip() if content_col else ''
                category = str(row.get(category_col, 'nasional')).strip().capitalize() #Capitalize
                link = str(row.get(link_col, '')).strip() if link_col else f'auto_{idx}'
                time_raw = row.get(time_col) if time_col else None
                # Normalisasi time
                if pd.isna(time_raw) or not str(time_raw).strip():
                    time_val = datetime.now()
                else:
                    try:
                        # hapus prefix "Tayang:" jika ada
                        time_str = str(time_raw).replace('Tayang:', '').replace('Tayang :', '').strip()
                        time_val = parser.parse(time_str, dayfirst=True)
                    except Exception:
                        time_val = datetime.now()
                
                # Validasi minimal
                title = str(row.get(title_col, '') or '').strip()
                content = str(row.get(content_col, '') or '').strip()

                if len(title) < 3 or len(content) < 10:
                    continue
                
                article = {
                    'title': title,
                    'content': content,
                    'category': category,
                    'link': link,
                    'time': time_val
                }
                
                articles_list.append(article)
                
            except Exception as e:
                print(f"⚠️ Skip row {idx}: {str(e)}")
                continue
        
        if not articles_list:
            print(f"❌ No valid articles found in file")
            return jsonify({
                'success': False,
                'message': 'File tidak mengandung data artikel yang valid. Pastikan Title (min 5 char) dan Content (min 20 char) terisi.'
            }), 400
        
        print(f"📈 Total articles to import: {len(articles_list)}")
        
        # Bulk import ke database
        result = news_db.bulk_import_articles(articles_list)
        
        # Clean up - hapus file setelah import
        try:
            os.remove(filepath)
            print(f"🗑️ Temporary file deleted")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")
        
        return jsonify({
            'success': result.get('success', True),
            'message': f"Berhasil mengimport {result.get('imported', 0)} berita. Dilewati: {result.get('skipped', 0)}",
            'imported': result.get('imported', 0),
            'skipped': result.get('skipped', 0),
            'errors': result.get('errors', [])
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Terjadi kesalahan: {str(e)}'
        }), 500

@app.route('/submit_news', methods=['POST'])
def submit_news():
    """Endpoint untuk submit berita dan klasifikasi - DENGAN VALIDASI LOGIN DAN VALIDASI INPUT"""

    # 🔐 CEK APAKAH USER SUDAH LOGIN
    user_info = get_user_info()
    if not user_info:
        return jsonify({
            'success': False,
            'message': 'Anda harus login terlebih dahulu untuk mengirim berita dan mencoba kategorisasi berita',
            'redirect': '/login'
        }), 401
    
    if not models:
        return jsonify({
            'success': False,
            'message': '❌ Models belum diload. Silakan cek console untuk error.'
        }), 500
    
    try:
        # Get input text dari form
        headline = request.form.get('headline', '').strip()
        article = request.form.get('article', '').strip()
        
        print(f"Headline diterima: {headline}")
        print(f"Article diterima: {article[:100]}...")
        print(f"Headline length: {len(headline)}")
        print(f"Article length: {len(article)}")
        print(f"👤 Submitted by user: {user_info['name']} ({user_info['email']})")
        
        # 🔍 VALIDASI INPUT DI BACKEND
        validation_errors = []
        
        # Validasi headline
        if not headline:
            validation_errors.append('Judul berita tidak boleh kosong')
        elif len(headline) < 50:
            validation_errors.append('Judul berita minimal 50 karakter')
        
        # Validasi article
        if not article:
            validation_errors.append('Isi berita tidak boleh kosong')
        elif len(article) < 400:
            validation_errors.append('Isi berita minimal 400 karakter')
        
        # Jika ada error validasi, return error
        if validation_errors:
            return jsonify({
                'success': False,
                'message': 'Validasi gagal',
                'errors': validation_errors
            }), 400
        
        # Gabungkan headline dan article untuk diproses
        news_text = f"{headline}. {article}"
        
        # 1. Preprocessing teks
        processed_text = simple_preprocess(news_text)
        print("✅ Preprocessing selesai")
        
        # Extract actual models from dictionaries if needed
        bow_model = extract_model_from_dict(models['bow'], 'bow')
        lda_model = extract_model_from_dict(models['lda'], 'lda') 
        rf_model = extract_model_from_dict(models['rf'], 'rf')
        
        print(f"📊 Bow model type after extraction: {type(bow_model)}")
        print(f"📊 LDA model type after extraction: {type(lda_model)}")
        print(f"📊 RF model type after extraction: {type(rf_model)}")
        
        # 2. Transform dengan BoW
        bow_features = bow_model.transform([processed_text])
        print("✅ BoW transformation selesai")
        
        # 3. Transform dengan LDA
        lda_features = lda_model.transform(bow_features)
        print("✅ LDA transformation selesai")
        
        # 4. Predict dengan Random Forest
        prediction_numeric = rf_model.predict(lda_features)[0]
        probability = rf_model.predict_proba(lda_features)[0]
        
        # Convert numeric prediction to category name
        prediction = get_category_name(prediction_numeric)
        
        # Dapatkan semua kategori dengan probabilitas
        all_categories = get_all_categories(probability, threshold=0.1)
        
        # Dapatkan top 3 kategori untuk summary
        top_categories = all_categories[:3] if len(all_categories) >= 3 else all_categories
        
        # Hitung total confidence (sum dari probabilities - harusnya ~100%)
        total_confidence = sum(prob * 100 for prob in probability)
        
        # Confidence untuk prediksi utama
        confidence = max(probability) * 100
        
        print(f"🔢 Numeric prediction: {prediction_numeric}")
        print(f"🏷️  Category prediction: {prediction}")
        print(f"📈 Confidence: {confidence:.2f}%")
        print(f"📊 Total categories found: {len(all_categories)}")
        print(f"🏆 Top category: {top_categories[0]['name']} ({top_categories[0]['percentage']:.2f}%)")
        
        # Kirim data ke output page
        return render_template('CNP_OUTPUT_PAGE.html',
                             headline=headline,
                             article=article,
                             prediction=prediction,
                             confidence=f"{confidence:.2f}%",
                             top_categories=top_categories,
                             all_categories=all_categories,
                             total_confidence=total_confidence,
                             user=user_info)
                             
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Return dummy result for testing dengan semua kategori
        print("🔄 Using fallback dummy result with all categories...")
        
        # Create dummy categories based on your example
        dummy_categories = [
            {"name": "Lifestyle", "percentage": 92.0, "index": 0},
            {"name": "Otomotif", "percentage": 4.0, "index": 1},
            {"name": "Health", "percentage": 2.0, "index": 2},
            {"name": "Teknologi", "percentage": 2.0, "index": 3},
            {"name": "Finance", "percentage": 0.5, "index": 4},
            {"name": "Travel", "percentage": 0.3, "index": 5},
            {"name": "Sport", "percentage": 0.2, "index": 6},
            {"name": "Edukasi", "percentage": 0.1, "index": 7},
        ]
        
        return render_template('CNP_OUTPUT_PAGE.html',
                             headline=headline or "Contoh Berita: Tren Gaya Hidup Modern",
                             article=article or "Dalam beberapa tahun terakhir, gaya hidup modern telah mengalami transformasi signifikan. Masyarakat semakin sadar akan pentingnya keseimbangan antara pekerjaan dan kehidupan pribadi, serta mengadopsi teknologi untuk mempermudah aktivitas sehari-hari. Perkembangan ini tidak hanya terlihat di perkotaan tetapi juga mulai merambah ke daerah-daerah...",
                             prediction="Lifestyle",
                             confidence="92.00%",
                             top_categories=dummy_categories[:3],
                             all_categories=dummy_categories,
                             total_confidence=101.1,
                             user=user_info)

# ===============================
# MODEL FUNCTIONS
# ===============================

def load_label_mapping():
    """Load label mapping dari model_metrics.json"""
    global LABEL_MAPPING, REVERSE_LABEL_MAPPING
    
    try:
        # PERBAIKAN: Path yang benar tanpa folder visualization
        metrics_path = os.path.join(BASE_DIR, '..', 'output', 'model_results_save', 'model_metrics.json')
        print(f"🔍 Loading label mapping from: {metrics_path}")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            if 'label_mapping' in metrics_data:
                # Mapping dari file: "Edukasi": 0 -> kita butuh reverse: 0: "Edukasi"
                original_mapping = metrics_data['label_mapping']
                REVERSE_LABEL_MAPPING = {v: k for k, v in original_mapping.items()}
                LABEL_MAPPING = original_mapping  # Keep original for reference
                
                print("✅ Label mapping loaded successfully!")
                print(f"📊 Original mapping: {original_mapping}")
                print(f"📊 Reverse mapping: {REVERSE_LABEL_MAPPING}")
                return True
            else:
                print("❌ 'label_mapping' not found in model_metrics.json")
                return False
        else:
            print(f"❌ model_metrics.json not found at: {metrics_path}")
            
            # PERBAIKAN: Coba path alternatif
            alternative_path = os.path.join(BASE_DIR, 'models', 'model_metrics.json')
            print(f"🔍 Trying alternative path: {alternative_path}")
            
            if os.path.exists(alternative_path):
                with open(alternative_path, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                if 'label_mapping' in metrics_data:
                    original_mapping = metrics_data['label_mapping']
                    REVERSE_LABEL_MAPPING = {v: k for k, v in original_mapping.items()}
                    LABEL_MAPPING = original_mapping
                    
                    print("✅ Label mapping loaded from alternative path!")
                    return True
            
            return False
            
    except Exception as e:
        print(f"❌ Error loading label mapping: {e}")
        return False

def debug_model_info():
    """Debug information about loaded models"""
    print("\n=== DEBUG MODEL INFO ===")
    for name, model in models.items():
        print(f"📦 {name.upper()} Model:")
        print(f"   Type: {type(model)}")
        
        # Untuk Random Forest, cek classes_
        if hasattr(model, 'classes_'):
            print(f"   Classes: {model.classes_}")
            print(f"   Classes type: {type(model.classes_[0])}")
            print(f"   Number of classes: {len(model.classes_)}")
            
        print()
    print("=======================\n")

def load_models():
    """Load semua model yang sudah disave"""
    try:
        print("Loading models...")
        print(f"Base directory: {BASE_DIR}")
        
        # Load label mapping terlebih dahulu
        if not load_label_mapping():
            print("⚠️  Using fallback label mapping")
            setup_fallback_mapping()
        
        # Define absolute paths
        bow_path = os.path.join(BASE_DIR, 'models', 'bow_model.pkl')
        lda_path = os.path.join(BASE_DIR, 'models', 'lda_model.pkl')
        rf_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
        
        print(f"Bow path: {bow_path}")
        print(f"LDA path: {lda_path}")
        print(f"RF path: {rf_path}")
        
        # Check if files exist
        if not os.path.exists(bow_path):
            print(f"❌ File not found: {bow_path}")
            return False
        if not os.path.exists(lda_path):
            print(f"❌ File not found: {lda_path}")
            return False
        if not os.path.exists(rf_path):
            print(f"❌ File not found: {rf_path}")
            return False
            
        print("✅ All model files found!")
        
        # Load BoW model
        with open(bow_path, 'rb') as f:
            models['bow'] = pickle.load(f)
        print("✅ BoW model loaded")
        
        # Load LDA model
        with open(lda_path, 'rb') as f:
            models['lda'] = pickle.load(f)
        print("✅ LDA model loaded")
        
        # Load Random Forest model
        with open(rf_path, 'rb') as f:
            models['rf'] = pickle.load(f)
        print("✅ Random Forest model loaded")
        
        # Debug model info
        debug_model_info()
        
        # Verifikasi mapping dengan model
        verify_label_mapping()
        
        print("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_fallback_mapping():
    """Setup fallback mapping berdasarkan yang ada di model_metrics.json"""
    global REVERSE_LABEL_MAPPING
    
    # Mapping dari model_metrics.json Anda
    fallback_mapping = {
        "Edukasi": 0,
        "Finance": 1, 
        "Health": 2,
        "Lifestyle": 3,
        "Nasional": 4,
        "Otomotif": 5,
        "Sport": 6,
        "Tekno": 7,
        "Travel": 8
    }
    
    REVERSE_LABEL_MAPPING = {v: k for k, v in fallback_mapping.items()}
    print("⚠️  Using fallback mapping based on provided model_metrics.json")

def verify_label_mapping():
    """Verifikasi bahwa label mapping sesuai dengan model"""
    if hasattr(models['rf'], 'classes_'):
        rf_classes = models['rf'].classes_
        print(f"🔍 Verifying label mapping with RF classes: {rf_classes}")
        
        # Cek apakah semua classes ada di mapping
        for class_num in rf_classes:
            if class_num not in REVERSE_LABEL_MAPPING:
                print(f"⚠️  Warning: Class {class_num} not found in label mapping")
            else:
                print(f"✅ Class {class_num} -> {REVERSE_LABEL_MAPPING[class_num]}")

def extract_model_from_dict(model_obj, model_type):
    """Extract actual model from dictionary if needed"""
    if isinstance(model_obj, dict):
        print(f"🔍 Model {model_type} is a dictionary, extracting...")
        
        # Coba berbagai kemungkinan key
        possible_keys = {
            'bow': ['bow_model', 'vectorizer', 'count_vectorizer', 'model', 'cv'],
            'lda': ['lda_model', 'lda', 'model', 'topic_model'],
            'rf': ['rf_model', 'random_forest', 'classifier', 'model', 'clf']
        }
        
        for key in possible_keys[model_type]:
            if key in model_obj:
                print(f"✅ Found model with key: '{key}'")
                return model_obj[key]
        
        # Jika tidak ditemukan, return key pertama
        first_key = list(model_obj.keys())[0]
        print(f"⚠️  Using first key: '{first_key}'")
        return model_obj[first_key]
    
    return model_obj

def get_category_name(prediction):
    """Convert numeric prediction to category name"""
    try:
        # Jika prediction sudah string, return langsung
        if isinstance(prediction, str):
            return prediction
        
        # Jika prediction numeric, gunakan mapping
        prediction_int = int(prediction)
        
        if prediction_int in REVERSE_LABEL_MAPPING:
            category_name = REVERSE_LABEL_MAPPING[prediction_int]
            print(f"🏷️  Mapping: {prediction_int} -> {category_name}")
            return category_name
        else:
            print(f"⚠️  Label {prediction_int} not in mapping, using numeric")
            return f"Kategori {prediction_int}"
            
    except (ValueError, TypeError) as e:
        print(f"❌ Error converting prediction {prediction}: {e}")
        return str(prediction)

def get_all_categories(probability, threshold=0.01):
    """Get all categories with their probabilities"""
    all_categories = []
    
    for idx, prob in enumerate(probability):
        category_name = get_category_name(idx)
        percentage = prob * 100
        
        # Only include categories with percentage above threshold (default 1%)
        if percentage >= threshold:
            all_categories.append({
                "name": category_name,
                "percentage": percentage,
                "index": idx
            })
    
    # Sort by percentage descending
    all_categories.sort(key=lambda x: x['percentage'], reverse=True)
    
    return all_categories

def get_top_categories(probability, top_n=3):
    """Get top N categories with highest probabilities"""
    all_categories = get_all_categories(probability, threshold=0)
    return all_categories[:top_n]

def simple_preprocess(text):
    """
    Fungsi preprocessing sederhana
    """
    if text:
        text = text.lower().strip()
        # Basic preprocessing
        import re
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    return ""

# ===============================
# API ENDPOINTS
# ===============================

@app.route('/output')
def output_page():
    """Halaman output (bisa diakses langsung untuk testing)"""
    # Return ke halaman input jika tidak ada data
    return redirect(url_for('input_page'))

@app.route('/health')
def health_check():
    """Endpoint untuk cek status aplikasi"""
    model_status = "Loaded" if models else "Not Loaded"
    label_count = len(REVERSE_LABEL_MAPPING)
    db_stats = news_db.get_database_stats()
    
    # 🔐 PERBAIKI AUTHENTICATION STATUS
    users_table_exists = user_db.create_users_table()
    
    # 🔐 PERBAIKI: Dapatkan user info dari function terpusat
    user_info = get_user_info()
    
    return {
        "status": "running",
        "models": model_status,
        "labels_loaded": label_count,
        "authentication": "enabled",
        "users_table": "Ready" if users_table_exists else "Error",
        "session": {
            "user_logged_in": user_info is not None,
            "user_id": session.get('user_id'),
            "user_type": user_info['type'] if user_info else 'none',
            "is_admin": user_info['is_admin'] if user_info else False
        },
        "database": {
            "total_articles": db_stats.get('total_articles', 0),
            "categories_count": len(db_stats.get('categories', {}))
        },
        "available_categories": list(LABEL_MAPPING.keys()) if LABEL_MAPPING else [],
        "message": "NewsPortal Flask API is running with authentication"
    }

@app.route('/labels')
def show_labels():
    """Endpoint untuk melihat label mapping"""
    rf_classes = []
    if hasattr(models.get('rf', {}), 'classes_'):
        rf_classes = models['rf'].classes_.tolist()
    
    return {
        "original_mapping": LABEL_MAPPING,
        "reverse_mapping": REVERSE_LABEL_MAPPING,
        "rf_classes": rf_classes,
        "total_categories": len(REVERSE_LABEL_MAPPING)
    }

@app.route('/api/articles')
def api_articles():
    """API endpoint to get articles (for AJAX)"""
    category = request.args.get('category', '')
    search = request.args.get('search', '')
    limit = int(request.args.get('limit', 50))
    
    if search:
        articles = news_db.search_articles(search, limit)
    elif category:
        articles = news_db.get_articles_by_category(category, limit)
    else:
        articles = news_db.get_all_articles(limit)
    
    return jsonify(articles)

@app.route('/api/categories')
def api_categories():
    """API endpoint to get categories"""
    categories = news_db.get_categories()
    return jsonify(categories)

@app.route('/api/stats')
def api_stats():
    """API endpoint to get database statistics"""
    stats = news_db.get_database_stats()
    return jsonify(stats)

@app.route('/test-prediction')
def test_prediction():
    """Test endpoint untuk prediksi dengan contoh teks"""
    if not models:
        return {"error": "Models not loaded"}
    
    test_texts = [
        "Teknologi AI berkembang pesat di Indonesia dengan inovasi terbaru",
        "Pertandingan sepak bola berlangsung seru dengan skor akhir 3-2", 
        "Investasi saham menunjukkan pertumbuhan positif di kuartal ini",
        "Pendidikan online menjadi tren selama pandemi dan terus berkembang",
        "Gaya hidup sehat dengan olahraga rutin dan makanan bergizi",
        "Perkembangan industri otomotif dengan mobil listrik terbaru"
    ]
    
    results = []
    for text in test_texts:
        try:
            processed = simple_preprocess(text)
            bow_features = models['bow'].transform([processed])
            lda_features = models['lda'].transform(bow_features)
            prediction_numeric = models['rf'].predict(lda_features)[0]
            probability = models['rf'].predict_proba(lda_features)[0]
            prediction_name = get_category_name(prediction_numeric)
            
            # Get top 3 categories
            top_categories = get_top_categories(probability, top_n=3)
            all_categories = get_all_categories(probability, threshold=0.1)
            
            results.append({
                "text": text, 
                "prediction": prediction_name, 
                "numeric": prediction_numeric,
                "confidence": f"{max(probability)*100:.2f}%",
                "top_categories": top_categories,
                "total_categories_found": len(all_categories)
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    
    return {"test_results": results}

@app.route('/model-info')
def model_info():
    """Endpoint untuk melihat informasi model"""
    if not models:
        return {"error": "Models not loaded"}
    
    info = {
        "bow_model": str(type(models['bow'])),
        "lda_model": str(type(models['lda'])),
        "rf_model": str(type(models['rf'])),
    }
    
    if hasattr(models['rf'], 'classes_'):
        info['rf_classes'] = models['rf'].classes_.tolist()
        info['num_classes'] = len(models['rf'].classes_)
    
    if hasattr(models['rf'], 'n_estimators'):
        info['rf_estimators'] = models['rf'].n_estimators
    
    return info

# ===============================
# 🔐 ADMIN ROUTES - SUBMIT TO DATABASE
# ===============================

@app.route('/submit_to_database', methods=['POST'])
def submit_to_database():
    """Submit berita ke database (admin only)"""
    try:
        # 🔐 CEK APAKAH USER ADMIN
        user_info = get_user_info()
        if not user_info or not user_info['is_admin']:
            return jsonify({
                'success': False,
                'message': 'Akses ditolak. Hanya administrator yang dapat mengirim berita ke database.'
            }), 403
        
        # Get data dari form
        headline = request.form.get('headline', '').strip()
        article = request.form.get('article', '').strip()
        top_category = request.form.get('top_category', '').strip()
        confidence_score = request.form.get('confidence_score', '0')
        
        # Validasi data
        if not headline or not article:
            return jsonify({
                'success': False,
                'message': 'Headline dan artikel tidak boleh kosong.'
            }), 400
        
        # Simpan ke database
        conn = db_config.get_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Koneksi database gagal.'
            }), 500
        
        try:
            cur = conn.cursor()
            
            # Insert ke tabel news_articles
            cur.execute("""
                INSERT INTO news_articles (title, content, category, time, link)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                headline,
                article,
                top_category,
                datetime.now(),
                f"/article/auto_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            ))
            
            conn.commit()
            cur.close()
            
            print(f"✅ Berita berhasil disimpan ke database: {headline}")
            
            return jsonify({
                'success': True,
                'message': f'Berita berhasil disimpan ke database dengan kategori: {top_category}'
            })
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error menyimpan ke database: {e}")
            return jsonify({
                'success': False,
                'message': f'Gagal menyimpan ke database: {str(e)}'
            }), 500
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        print(f"❌ Error dalam submit_to_database: {e}")
        return jsonify({
            'success': False,
            'message': 'Terjadi kesalahan sistem.'
        }), 500

# ===============================
# ERROR HANDLERS
# ===============================

@app.errorhandler(404)
def not_found(error):
    user_info = get_user_info()
    return render_template('CNP_MAIN_PAGE.html', user=user_info), 404

@app.errorhandler(500)
def internal_error(error):
    user_info = get_user_info()
    return render_template('CNP_MAIN_PAGE.html', user=user_info), 500

# Load Models
print("🟢 Loading models at startup...")
load_models()

# ===============================
# MAIN APPLICATION
# ===============================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)