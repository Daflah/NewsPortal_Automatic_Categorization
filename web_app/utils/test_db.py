import psycopg2

try:
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        database="newsportal",
        user="postgres",
        password="24Dappe08"  # ganti dengan password Anda
    )
    
    # Create cursor
    cur = conn.cursor()
    
    # Test connection
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print("✅ PostgreSQL connection successful!")
    print(f"📊 PostgreSQL version: {version[0]}")
    
    # Close connection
    cur.close()
    conn.close()
    
except Exception as error:
    print(f"❌ Error connecting to PostgreSQL: {error}")