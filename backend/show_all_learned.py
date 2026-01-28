"""Show ALL entries in Learning Agent collection"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

results = collection.get()

print(f"Total entries: {len(results['ids'])}\n")
print("=" * 100)

for i, (doc, meta, id_) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
    print(f"\n{i+1}. ID: {id_}")
    print(f"   Document (first 150 chars): {doc[:150]}")
    print(f"   Selector: {meta.get('selector', 'N/A')}")
    print(f"   Ticket: {meta.get('ticket_id', 'N/A')}")

    # Show if it contains delete button
    if 'deletebtn' in meta.get('selector', '').lower() or 'deletebtn' in doc.lower():
        print(f"   *** CONTAINS DELETE BUTTON ***")
