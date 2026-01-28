"""Find delete button entries in Learning Agent by searching documents"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

results = collection.get()

print(f"Total entries: {len(results['ids'])}\n")
print("Searching for 'delete' or 'default_testobject_01'...\n")
print("=" * 100)

for i, (doc, meta, id_) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
    if 'default_testobject_01' in doc.lower() or 'default_testobject_01' in meta.get('selector', ''):
        print(f"\n{i+1}. ID: {id_}")
        print(f"   Document: {doc[:200]}")
        print(f"   Selector: {meta.get('selector', 'N/A')}")
        print(f"   Ticket: {meta.get('ticket_id', 'N/A')}")
        print(f"   Source: {meta.get('source', 'N/A')}")
