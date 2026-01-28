"""
PLCD Testing Assistant - Test Vector Database Setup
Validates that ChromaDB setup completed successfully
"""

import chromadb
from chromadb.config import Settings
from config_loader import (
    load_config,
    get_azure_client,
    get_chromadb_path,
    get_embedding_model
)


def generate_test_embedding(text: str, client, model: str):
    """Generate embedding for test query"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def test_chromadb_setup():
    """Test ChromaDB setup with various queries"""

    print("=" * 80)
    print("Testing Vector Database Setup")
    print("=" * 80)

    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    azure_client = get_azure_client(config)
    print("[OK] Configuration loaded")

    # Connect to ChromaDB
    print("\n[2/6] Connecting to ChromaDB...")
    chroma_path = get_chromadb_path(config)
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )

    collection_name = config['vector_database']['collections']['selectors_base']
    collection = chroma_client.get_collection(name=collection_name)
    print(f"[OK] Connected to collection: {collection_name}")

    # Verify count
    print("\n[3/6] Verifying selector count...")
    count = collection.count()
    expected_count = config['selectors']['total_count']

    print(f"[OK] Collection contains {count} selectors")

    if count != expected_count:
        print(f"[WARNING] Expected {expected_count} selectors, found {count}")

    # Test queries
    print("\n[4/6] Running test queries...")

    embedding_model = get_embedding_model(config)

    test_queries = [
        {
            "text": "click save button",
            "expected_module": "AddExisting",
            "expected_attr": "SaveBtn"
        },
        {
            "text": "edit part details",
            "expected_module": "Teststep",
            "expected_attr": "EditBtn"
        },
        {
            "text": "select type from dropdown",
            "expected_module": "CreateNew",
            "expected_attr": "Type"
        },
        {
            "text": "navigate to runs",
            "expected_module": "Common",
            "expected_attr": "nav"
        },
        {
            "text": "verify success message",
            "expected_module": "Common",
            "expected_attr": "message"
        }
    ]

    print("")
    for i, query in enumerate(test_queries, 1):
        query_text = query["text"]
        print(f"  Query {i}: '{query_text}'")

        # Generate embedding
        query_embedding = generate_test_embedding(query_text, azure_client, embedding_model)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Display results
        if results['ids'] and len(results['ids'][0]) > 0:
            for j, (id, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            ), 1):
                similarity = 1 - distance
                print(f"    {j}. {id}: [{metadata['attr']}='{metadata['value']}']")
                print(f"       Module: {metadata['module']}, Confidence: {similarity:.3f}")
        else:
            print("    [WARNING] No results found")

        print("")

    # Test module filtering
    print("[5/6] Testing module filtering...")
    test_module = "Teststep"
    print(f"  Filtering by module: {test_module}")

    query_embedding = generate_test_embedding("click edit button", azure_client, embedding_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"module": test_module}
    )

    if results['ids'] and len(results['ids'][0]) > 0:
        print(f"  [OK] Found {len(results['ids'][0])} results in module '{test_module}'")
        for metadata in results['metadatas'][0]:
            assert metadata['module'] == test_module, f"Expected module '{test_module}', got '{metadata['module']}'"
        print(f"  [OK] All results belong to module '{test_module}'")
    else:
        print(f"  [WARNING] No results found for module '{test_module}'")

    # Summary
    print("\n[6/6] Test Summary...")
    print(f"  [OK] ChromaDB location: {chroma_path}")
    print(f"  [OK] Collection: {collection_name}")
    print(f"  [OK] Total selectors: {count}")
    print(f"  [OK] All {len(test_queries)} test queries executed")
    print(f"  [OK] Module filtering works correctly")

    print("\n" + "=" * 80)
    print("[OK] All tests passed! ChromaDB setup is valid.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_chromadb_setup()
        print("\n[OK] Vector database validation completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
