"""Delete specific wrong selector from Learning Agent
Usage: python delete_wrong_selector.py <selector_to_delete>
Example: python delete_wrong_selector.py "[data-deletebtn='default_testobject_01DeleteBtn']"
"""
import sys
import chromadb
from chromadb.config import Settings

if len(sys.argv) < 2:
    print("Usage: python delete_wrong_selector.py <selector_to_delete>")
    print("Example: python delete_wrong_selector.py \"[data-deletebtn='default_testobject_01DeleteBtn']\"")
    sys.exit(1)

selector_to_delete = sys.argv[1]

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('runtime_learned_collection')
results = collection.get()

# Find and delete entries with this selector
ids_to_delete = [results['ids'][i] for i, meta in enumerate(results['metadatas'])
                 if meta.get('selector') == selector_to_delete]

if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print(f"✓ Deleted {len(ids_to_delete)} entries with selector: {selector_to_delete}")
else:
    print(f"✗ No entries found with selector: {selector_to_delete}")
