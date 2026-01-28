"""Check what's in Learning Agent collection"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

results = collection.get()
print(f"Total learned entries: {len(results['ids'])}\n")

for i, (doc, meta, id_) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
    print(f"{i+1}. ID: {id_}")
    print(f"   Doc: {doc[:100]}...")
    print(f"   Selector: {meta.get('selector', 'N/A')}")
    print(f"   Ticket: {meta.get('ticket_id', 'N/A')}")
    print()
