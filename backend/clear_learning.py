"""Clear the entire Learning Agent collection"""
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='./data/chromadb', settings=Settings(anonymized_telemetry=False))
client.delete_collection('runtime_learned_collection')
print('✓ Cleared learning collection - all learned selectors removed')
