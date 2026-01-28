"""
Upload Tickets from Jira_Tickets Folder to Database
Scans your Jira_Tickets folder and adds all ticket files to the database

Usage:
    python upload_tickets_from_folder.py
"""

import psycopg2
from pathlib import Path
from datetime import datetime
import re

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'test_automation',
    'user': 'postgres',
    'password': 'postgres123'
}

# Jira_Tickets folder path
JIRA_TICKETS_FOLDER = Path("Jira_Tickets")


def extract_ticket_id_from_filename(filename):
    """
    Extract ticket ID from filename
    Examples:
        - RBPLCD-8835.txt -> RBPLCD-8835
        - RBPLCD-8835_report.html -> RBPLCD-8835
        - TEST-2024-001.txt -> TEST-2024-001
    """
    # Try common patterns
    patterns = [
        r'(RBPLCD-\d+)',           # RBPLCD-8835
        r'(TEST-\d+-\d+)',         # TEST-2024-001
        r'([A-Z]+-\d+)',           # Generic JIRA pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # Fallback: use filename without extension
    return Path(filename).stem


def extract_title_from_content(content, ticket_id):
    """
    Try to extract title from file content
    """
    try:
        # Try HTML title tag
        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try first heading
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE)
        if h1_match:
            return re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
        
        # Try first non-empty line for .txt files
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                return line[:200]  # Limit length
        
    except:
        pass
    
    # Fallback to ticket_id as title
    return f"Test Case - {ticket_id}"


def determine_module_from_content(content):
    """
    Try to determine module from content keywords
    """
    content_lower = content.lower()
    
    modules = {
        'authentication': ['login', 'logout', 'sign in', 'sign up', 'password', 'user credentials'],
        'dashboard': ['dashboard', 'home page', 'main page'],
        'measurement': ['measurement', 'measure', 'test step', 'teststep'],
        'user management': ['user profile', 'user management', 'account settings'],
        'reporting': ['report', 'generate report', 'export'],
    }
    
    for module, keywords in modules.items():
        if any(keyword in content_lower for keyword in keywords):
            return module.title()
    
    return "General"


def scan_and_upload_tickets():
    """
    Scan Jira_Tickets folder and upload all ticket files to database
    """
    print("=" * 80)
    print("📂 UPLOAD TICKETS FROM JIRA_TICKETS FOLDER")
    print("=" * 80)
    
    # Check if folder exists
    if not JIRA_TICKETS_FOLDER.exists():
        print(f"❌ Folder '{JIRA_TICKETS_FOLDER}' not found!")
        print(f"   Current directory: {Path.cwd()}")
        print("\n   Please create the folder and add ticket files.")
        return False
    
    print(f"✅ Found folder: {JIRA_TICKETS_FOLDER.absolute()}")
    
    # Get all files in folder
    all_files = list(JIRA_TICKETS_FOLDER.glob("*"))
    
    # Filter for valid ticket files (exclude subdirectories)
    ticket_files = [
        f for f in all_files 
        if f.is_file() and f.suffix.lower() in ['.txt', '.html', '.htm']
    ]
    
    if not ticket_files:
        print(f"\n⚠️  No ticket files found in {JIRA_TICKETS_FOLDER}")
        print("   Supported formats: .txt, .html, .htm")
        print(f"\n   Files in folder: {[f.name for f in all_files]}")
        return False
    
    print(f"\n📋 Found {len(ticket_files)} ticket file(s):")
    for f in ticket_files:
        print(f"   • {f.name}")
    
    # Connect to database
    print("\n📡 Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # Check/Create default project
    print("\n🏢 Checking default project...")
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
        print(f"⚠️  Could not verify project: {e}")
    
    # Process each ticket file
    print("\n📤 Uploading tickets to database...")
    print("-" * 80)
    
    added_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for ticket_file in ticket_files:
        try:
            # Extract ticket ID from filename
            ticket_id = extract_ticket_id_from_filename(ticket_file.name)
            
            # Read file content
            try:
                with open(ticket_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(ticket_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Extract metadata from content
            title = extract_title_from_content(content, ticket_id)
            module = determine_module_from_content(content)
            
            # Get absolute file path
            file_path = str(ticket_file.absolute())
            
            # Check if ticket already exists
            cursor.execute("SELECT id FROM tickets WHERE ticket_id = %s", (ticket_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing ticket
                cursor.execute("""
                    UPDATE tickets 
                    SET title = %s, module = %s, file_path = %s
                    WHERE ticket_id = %s
                """, (title, module, file_path, ticket_id))
                
                print(f"   🔄 Updated: {ticket_id}")
                print(f"      Title: {title}")
                print(f"      Module: {module}")
                print(f"      File: {ticket_file.name}")
                updated_count += 1
            else:
                # Insert new ticket
                cursor.execute("""
                    INSERT INTO tickets (ticket_id, title, module, project_id, file_path, created_at)
                    VALUES (%s, %s, %s, 1, %s, %s)
                """, (ticket_id, title, module, file_path, datetime.now()))
                
                print(f"   ✅ Added: {ticket_id}")
                print(f"      Title: {title}")
                print(f"      Module: {module}")
                print(f"      File: {ticket_file.name}")
                added_count += 1
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error processing {ticket_file.name}: {e}")
            error_count += 1
            print()
    
    # Commit all changes
    conn.commit()
    
    # Summary
    print("=" * 80)
    print("📊 UPLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Tickets added:    {added_count}")
    print(f"🔄 Tickets updated:  {updated_count}")
    print(f"❌ Errors:           {error_count}")
    
    # Show all tickets in database
    print("\n📋 All tickets in database:")
    print("-" * 80)
    cursor.execute("""
        SELECT ticket_id, title, module, file_path 
        FROM tickets 
        ORDER BY created_at DESC
    """)
    
    all_tickets = cursor.fetchall()
    for i, (ticket_id, title, module, file_path) in enumerate(all_tickets, 1):
        file_name = Path(file_path).name if file_path else 'N/A'
        print(f"{i}. {ticket_id}")
        print(f"   Title: {title}")
        print(f"   Module: {module}")
        print(f"   File: {file_name}")
        print()
    
    print("=" * 80)
    print(f"📊 Total tickets in database: {len(all_tickets)}")
    print("=" * 80)
    
    print("\n✅ Upload complete! You can now:")
    print("   1. Start FastAPI server: python main.py")
    print("   2. List tickets: GET http://localhost:8000/api/tickets")
    print("   3. Execute test: POST http://localhost:8000/api/execute-test?ticket_id=RBPLCD-8835")
    print()
    
    cursor.close()
    conn.close()
    return True


def list_files_in_folder():
    """
    Just list what files are in the Jira_Tickets folder
    """
    print("\n📂 Checking Jira_Tickets folder contents...")
    print(f"   Path: {JIRA_TICKETS_FOLDER.absolute()}")
    
    if not JIRA_TICKETS_FOLDER.exists():
        print(f"   ❌ Folder does not exist!")
        return
    
    all_items = list(JIRA_TICKETS_FOLDER.glob("*"))
    
    if not all_items:
        print(f"   ⚠️  Folder is empty")
        return
    
    print(f"\n   Found {len(all_items)} item(s):")
    for item in all_items:
        icon = "📄" if item.is_file() else "📁"
        size = f"({item.stat().st_size} bytes)" if item.is_file() else ""
        print(f"   {icon} {item.name} {size}")


if __name__ == "__main__":
    try:
        # First show what's in the folder
        list_files_in_folder()
        
        print("\n" + "=" * 80)
        
        # Ask for confirmation
        response = input("\n🚀 Ready to upload tickets to database? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            scan_and_upload_tickets()
        else:
            print("\n❌ Upload cancelled")
            
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()