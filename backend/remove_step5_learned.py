"""Remove Step 5 learned selector for RBPLCD-8835_1"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

# Delete all RBPLCD-8835_1 entries (new ticket, forces relearning with action keyword fix)
results = collection.get()
ids_to_delete = [id_ for id_, meta in zip(results['ids'], results['metadatas']) if meta.get('ticket_id') == 'RBPLCD-8835_1']

if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} entries for RBPLCD-8835_1")
else:
    print("No RBPLCD-8835_1 entries found")
