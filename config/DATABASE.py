import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables dari file .env di folder config
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'aws-1-ap-northeast-1.pooler.supabase.com') 
        self.database = os.getenv('DB_NAME', 'newsportal')
        self.user = os.getenv('DB_USER', 'postgres.zdguxjnmjmcojtirsxns')
        self.password = os.getenv('DB_PASSWORD', '24Dapadap08')
        self.port = os.getenv('DB_PORT', '6543')  
    
    def get_connection(self):
        """Membuat koneksi ke database Supabase"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                connect_timeout=10,
                options='-c statement_timeout=30000'
            )
            
            print("✅ Database connection successful!")
            return conn
            
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return None

# Instance untuk digunakan
db_config = DatabaseConfig()