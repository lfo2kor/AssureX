"""Delete all entries with delete button selector"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')

results = collection.get()

# Find all entries with 'deletebtn' in selector
ids_to_delete = []
for i, meta in enumerate(results['metadatas']):
    selector = meta.get('selector', '')
    if 'deletebtn' in selector.lower() or 'default_testobject_01DeleteBtn' in selector:
        ids_to_delete.append(results['ids'][i])
        print(f"Will delete: {selector}")

if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print(f"\n✓ Deleted {len(ids_to_delete)} delete button entries")
else:
    print("✗ No delete button entries found")
