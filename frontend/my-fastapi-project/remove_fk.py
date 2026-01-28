import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()
    
    print("🔧 Removing foreign key constraint...")
    cursor.execute("""
        ALTER TABLE test_executions 
        DROP CONSTRAINT IF EXISTS test_executions_ticket_id_fkey;
    """)
    
    conn.commit()
    print("✅ Done!")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")