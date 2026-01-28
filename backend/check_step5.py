"""Check Step 5 entries in Learning Agent"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

results = collection.get()
print("Step 5 entries (edit button):\n")

for i, (doc, meta, id_) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
    if 'edit' in doc.lower() and 'default_testobject_01' in doc.lower():
        print(f"ID: {id_}")
        print(f"Doc: {doc[:150]}")
        print(f"Selector: {meta.get('selector', 'N/A')}")
        print(f"Ticket: {meta.get('ticket_id', 'N/A')}")
        print("-" * 80)
