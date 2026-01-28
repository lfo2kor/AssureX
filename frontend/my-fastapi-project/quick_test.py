"""
Quick test script to verify database setup and add test tickets
Run this first to quickly set up your database with sample tickets

Usage:
    python quick_test.py
"""

import psycopg2
from datetime import datetime

# Your database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'test_automation',
    'user': 'postgres',
    'password': 'postgres123'
}


def test_connection_and_add_tickets():
    """Test database and add sample tickets"""
    
    print("=" * 80)
    print("🚀 Quick Database Setup & Test")
    print("=" * 80)
    
    # Step 1: Test connection
    print("\n📡 Step 1: Testing database connection...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to PostgreSQL successfully!")
        cursor = conn.cursor()
        
        # Check database
        cursor.execute("SELECT current_database()")
        db_name = cursor.fetchone()[0]
        print(f"✅ Current database: {db_name}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database 'test_automation' exists")
        print("3. Username/password are correct")
        return False
    
    # Step 2: Check tables
    print("\n📋 Step 2: Checking required tables...")
    try:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('tickets', 'test_executions', 'projects', 'execution_steps')
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['tickets', 'test_executions', 'projects', 'execution_steps']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            print(f"⚠️  Missing tables: {missing_tables}")
            print("\n   Solution: Run your FastAPI app once to create tables:")
            print("   python main.py")
            return False
        
        print(f"✅ All required tables exist: {', '.join(tables)}")
        
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False
    
    # Step 3: Check/Create default project
    print("\n🏢 Step 3: Checking default project...")
    try:
        cursor.execute("SELECT id, name FROM projects WHERE id = 1")
        project = cursor.fetchone()
        
        if not project:
            print("⚠️  Default project not found, creating it...")
            cursor.execute("""
                INSERT INTO projects (id, name, base_folder, created_at)
                VALUES (1, 'Default Project', '/default', %s)
            """, (datetime.now(),))
            conn.commit()
            print("✅ Created default project (ID: 1)")
        else:
            print(f"✅ Default project exists: {project[1]}")
            
    except Exception as e:
        print(f"⚠️  Could not create project: {e}")
        # Continue anyway
    
    # Step 4: Check existing tickets
    print("\n🎫 Step 4: Checking existing tickets...")
    try:
        cursor.execute("SELECT COUNT(*) FROM tickets")
        ticket_count = cursor.fetchone()[0]
        print(f"📊 Current tickets in database: {ticket_count}")
        
        if ticket_count > 0:
            cursor.execute("SELECT ticket_id, title FROM tickets ORDER BY created_at DESC LIMIT 5")
            existing_tickets = cursor.fetchall()
            print("\n   Recent tickets:")
            for ticket_id, title in existing_tickets:
                print(f"   - {ticket_id}: {title}")
        
    except Exception as e:
        print(f"❌ Error checking tickets: {e}")
    
    # Step 5: Add sample tickets
    print("\n➕ Step 5: Adding sample tickets...")
    
    sample_tickets = [
        ("RBPLCD-8835", "Test Measurement Configuration", "Measurement"),
        ("RBPLCD-1001", "Login Functionality Test", "Authentication"),
        ("RBPLCD-1002", "Dashboard UI Test", "Dashboard"),
        ("TEST-2024-001", "Sample Test Case", "General"),
        ("RBPLCD-2025", "User Profile Update Test", "User Management"),
    ]
    
    added_count = 0
    skipped_count = 0
    
    for ticket_id, title, module in sample_tickets:
        try:
            # Check if exists
            cursor.execute("SELECT id FROM tickets WHERE ticket_id = %s", (ticket_id,))
            if cursor.fetchone():
                print(f"   ⏭️  Skipped {ticket_id} (already exists)")
                skipped_count += 1
                continue
            
            # Insert
            cursor.execute("""
                INSERT INTO tickets (ticket_id, title, module, project_id, created_at)
                VALUES (%s, %s, %s, 1, %s)
            """, (ticket_id, title, module, datetime.now()))
            
            print(f"   ✅ Added {ticket_id}: {title}")
            added_count += 1
            
        except Exception as e:
            print(f"   ❌ Error adding {ticket_id}: {e}")
    
    conn.commit()
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Database connection: OK")
    print(f"✅ Tables: OK")
    print(f"✅ Default project: OK")
    print(f"📝 Tickets added: {added_count}")
    print(f"⏭️  Tickets skipped: {skipped_count}")
    
    # Show final count
    cursor.execute("SELECT COUNT(*) FROM tickets")
    final_count = cursor.fetchone()[0]
    print(f"📊 Total tickets in database: {final_count}")
    
    # Show all tickets
    print("\n📋 All tickets:")
    cursor.execute("SELECT ticket_id, title, module FROM tickets ORDER BY created_at DESC")
    all_tickets = cursor.fetchall()
    for ticket_id, title, module in all_tickets:
        print(f"   • {ticket_id} - {title} ({module})")
    
    print("\n" + "=" * 80)
    print("✅ Setup complete! You can now:")
    print("   1. Start your FastAPI server: python main.py")
    print("   2. Test execution: POST /api/execute-test?ticket_id=RBPLCD-8835")
    print("   3. List tickets: GET /api/tickets")
    print("=" * 80)
    
    cursor.close()
    conn.close()
    return True


if __name__ == "__main__":
    try:
        test_connection_and_add_tickets()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()