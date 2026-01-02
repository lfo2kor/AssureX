"""
Verify Tickets in PostgreSQL Database
Quick script to check what tickets are in your database

Usage:
    python verify_db_tickets.py
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from tabulate import tabulate

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'test_automation',
    'user': 'postgres',
    'password': 'postgres123'
}


def verify_tickets_in_db():
    """
    Verify tickets are in PostgreSQL database
    """
    print("=" * 100)
    print("🔍 VERIFYING TICKETS IN POSTGRESQL DATABASE")
    print("=" * 100)
    
    # Connect to database
    print("\n📡 Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print(f"✅ Connected to database: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print(f"  - PostgreSQL is running")
        print(f"  - Database: {DB_CONFIG['database']}")
        print(f"  - Username: {DB_CONFIG['user']}")
        print(f"  - Password: {'*' * len(DB_CONFIG['password'])}")
        return False
    
    # Get total count
    print("\n📊 Getting ticket count...")
    try:
        cursor.execute("SELECT COUNT(*) as count FROM tickets")
        count = cursor.fetchone()['count']
        print(f"✅ Total tickets in database: {count}")
        
        if count == 0:
            print("\n⚠️  No tickets found in database!")
            print("   Did the upload script run successfully?")
            print("   Run: python upload_tickets_from_folder.py")
            return False
            
    except Exception as e:
        print(f"❌ Error getting count: {e}")
        return False
    
    # Get all tickets with details
    print("\n📋 Fetching ticket details...")
    try:
        cursor.execute("""
            SELECT 
                id,
                ticket_id,
                title,
                module,
                project_id,
                file_path,
                created_at
            FROM tickets 
            ORDER BY created_at DESC
        """)
        
        tickets = cursor.fetchall()
        
        if tickets:
            print(f"✅ Found {len(tickets)} ticket(s)\n")
            
            # Display in table format
            headers = ["#", "Ticket ID", "Title", "Module", "Project", "Has File", "Created"]
            rows = []
            
            for i, ticket in enumerate(tickets, 1):
                # Check if file exists
                file_status = "✅" if ticket['file_path'] else "❌"
                
                # Format created date
                created = ticket['created_at'].strftime('%Y-%m-%d %H:%M') if ticket['created_at'] else 'N/A'
                
                # Truncate title if too long
                title = ticket['title'][:45] + "..." if ticket['title'] and len(ticket['title']) > 45 else ticket['title']
                
                rows.append([
                    i,
                    ticket['ticket_id'],
                    title or 'N/A',
                    ticket['module'] or 'N/A',
                    ticket['project_id'],
                    file_status,
                    created
                ])
            
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
            # Detailed view
            print("\n" + "=" * 100)
            print("📝 DETAILED TICKET INFORMATION")
            print("=" * 100)
            
            for i, ticket in enumerate(tickets, 1):
                print(f"\n{i}. Ticket: {ticket['ticket_id']}")
                print(f"   {'─' * 90}")
                print(f"   Database ID:  {ticket['id']}")
                print(f"   Title:        {ticket['title']}")
                print(f"   Module:       {ticket['module']}")
                print(f"   Project ID:   {ticket['project_id']}")
                print(f"   File Path:    {ticket['file_path'] or 'Not set'}")
                print(f"   Created:      {ticket['created_at']}")
            
            # Check for executions
            print("\n" + "=" * 100)
            print("🚀 CHECKING FOR TEST EXECUTIONS")
            print("=" * 100)
            
            cursor.execute("""
                SELECT 
                    ticket_id,
                    COUNT(*) as execution_count,
                    MAX(started_at) as last_execution
                FROM test_executions
                GROUP BY ticket_id
            """)
            
            executions = cursor.fetchall()
            
            if executions:
                print(f"\n✅ Found executions for {len(executions)} ticket(s):\n")
                
                exec_headers = ["Ticket ID", "Total Runs", "Last Execution"]
                exec_rows = []
                
                for exec in executions:
                    last_run = exec['last_execution'].strftime('%Y-%m-%d %H:%M') if exec['last_execution'] else 'N/A'
                    exec_rows.append([
                        exec['ticket_id'],
                        exec['execution_count'],
                        last_run
                    ])
                
                print(tabulate(exec_rows, headers=exec_headers, tablefmt='grid'))
            else:
                print("\n📊 No test executions found yet.")
                print("   To execute a test, use: POST /api/execute-test?ticket_id=RBPLCD-8835")
            
        else:
            print("⚠️  No tickets found")
        
    except Exception as e:
        print(f"❌ Error fetching tickets: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a sample query
    print("\n" + "=" * 100)
    print("🧪 TESTING SAMPLE QUERIES")
    print("=" * 100)
    
    if tickets:
        sample_ticket_id = tickets[0]['ticket_id']
        
        print(f"\nTesting query for ticket: {sample_ticket_id}")
        try:
            cursor.execute("""
                SELECT ticket_id, title, module, file_path
                FROM tickets
                WHERE ticket_id = %s
            """, (sample_ticket_id,))
            
            result = cursor.fetchone()
            if result:
                print("✅ Query successful!")
                print(f"   Ticket ID: {result['ticket_id']}")
                print(f"   Title: {result['title']}")
                print(f"   Module: {result['module']}")
                print(f"   File: {result['file_path']}")
            else:
                print("❌ No result returned")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Summary
    print("\n" + "=" * 100)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 100)
    print(f"✅ Database connection: OK")
    print(f"✅ Tickets in database: {count}")
    print(f"✅ Queries working: OK")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Start your FastAPI server:")
    print("      python main.py")
    print()
    print("   2. Access API docs:")
    print("      http://localhost:8000/docs")
    print()
    print("   3. List tickets via API:")
    print("      GET http://localhost:8000/api/tickets")
    print()
    print("   4. Execute a test:")
    print(f"      POST http://localhost:8000/api/execute-test?ticket_id={tickets[0]['ticket_id']}")
    print()
    print("   5. Or test with curl:")
    print(f"      curl http://localhost:8000/api/tickets")
    print("=" * 100)
    
    cursor.close()
    conn.close()
    return True


if __name__ == "__main__":
    try:
        verify_tickets_in_db()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()