"""
PLCD Testing Assistant - Agent 1: Selector Discovery (L1)
Performs semantic search in ChromaDB to find best matching selectors
"""

import logging
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings

from config_loader import (
    load_config,
    get_azure_client,
    get_chromadb_path,
    get_embedding_model
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/agent1_selector_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Agent1SelectorDiscovery:
    """
    Agent 1: Semantic Selector Discovery

    Finds best matching selector from ChromaDB using semantic search
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Agent 1

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.azure_client = get_azure_client(config)
        self.embedding_model = get_embedding_model(config)

        # Connect to ChromaDB
        chroma_path = get_chromadb_path(config)
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        collection_name = config['vector_database']['collections']['selectors_base']
        self.collection = self.chroma_client.get_collection(name=collection_name)

        # Load agent configuration
        self.agent_config = config['agent1_selector_discovery']

        logger.debug(f"Agent 1 initialized with collection: {collection_name}")
        logger.debug(f"Collection contains {self.collection.count()} selectors")


    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Azure OpenAI

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.azure_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding


    def query_selectors(
        self,
        step_text: str,
        current_module: str,
        page_context: str = None,
        n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB for matching selectors

        Args:
            step_text: Test step description
            current_module: Current UI module
            page_context: Current page context for filtering
            n_results: Number of results to return (default from config)

        Returns:
            ChromaDB query results
        """
        # Generate embedding for step
        logger.debug(f"Generating embedding for: '{step_text}'")
        step_embedding = self.generate_embedding(step_text)

        # Build module filter (current module + common modules)
        module_filter = [current_module] + self.config['modules']['common_modules']
        logger.debug(f"Module filter: {module_filter}")

        # Get n_results from config if not specified
        if n_results is None:
            n_results = self.agent_config['retrieval']['n_results']

        # Build metadata filter
        if page_context:
            # Use $and for compound filter
            where_filter = {
                "$and": [
                    {"module": {"$in": module_filter}},
                    {"page_context": page_context}
                ]
            }
            logger.debug(f"Page context filter: {page_context}")
        else:
            # Simple module filter only
            where_filter = {"module": {"$in": module_filter}}

        # Query ChromaDB with metadata filters
        try:
            results = self.collection.query(
                query_embeddings=[step_embedding],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            logger.warning("Returning empty result due to ChromaDB error - system will fall back to L2")
            # Return empty result structure (system will fall back to L2/L3)
            return {
                'ids': [[]],
                'distances': [[]],
                'metadatas': [[]],
                'documents': [[]]
            }


    def calculate_confidence(
        self,
        distance: float,
        metadata: Dict[str, Any],
        current_module: str
    ) -> float:
        """
        Calculate composite confidence score

        Args:
            distance: Cosine distance from ChromaDB (0 = perfect match)
            metadata: Selector metadata
            current_module: Current UI module

        Returns:
            Confidence score (0.0 - 1.0)
        """
        # Convert distance to similarity (cosine similarity = 1 - cosine distance)
        similarity = 1.0 - distance

        # Module match bonus
        module_match = 1.0 if metadata.get('module') == current_module else 0.5

        # Priority score (normalize 0-100 to 0-1)
        priority_score = metadata.get('priority', 50) / 100.0

        # Calculate weighted confidence
        scoring_config = self.agent_config['scoring']
        confidence = (
            similarity * scoring_config['semantic_similarity_weight'] +
            module_match * scoring_config['module_match_weight'] +
            priority_score * scoring_config['priority_weight']
        )

        # Cap at 1.0
        return min(confidence, 1.0)


    def discover_selector(
        self,
        step_text: str,
        current_module: str,
        page_context: str = None,
        n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Discover best selector for test step

        Args:
            step_text: Test step description
            current_module: Current UI module
            page_context: Current page context for metadata filtering
            n_results: Number of results to return (default from config)

        Returns:
            Dictionary containing:
            - selector_result: Best matching selector with confidence
            - candidates: All candidates with scores
        """
        logger.debug("=" * 80)
        logger.debug(f"Agent 1: Discovering selector")
        logger.debug(f"Step: {step_text}")
        logger.debug(f"Module: {current_module}")
        if page_context:
            logger.debug(f"Page context: {page_context}")

        # Query ChromaDB
        results = self.query_selectors(step_text, current_module, page_context, n_results)

        # Build candidates list
        candidates = []

        if not results['ids'] or len(results['ids'][0]) == 0:
            logger.warning("No selectors found in ChromaDB")
            return {
                "selector_result": None,
                "candidates": []
            }

        for i in range(len(results['ids'][0])):
            selector_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]

            # Calculate confidence
            confidence = self.calculate_confidence(distance, metadata, current_module)

            candidate = {
                "selector": metadata.get('full_selector', ''),
                "confidence": confidence,
                "agent_used": "L1",
                "metadata": {
                    "id": selector_id,
                    "attr": metadata.get('attr', ''),
                    "value": metadata.get('value', ''),
                    "module": metadata.get('module', ''),
                    "elementType": metadata.get('elementType', ''),
                    "label": metadata.get('label', ''),
                    "priority": metadata.get('priority', 50),
                    "isDynamic": metadata.get('isDynamic', False),
                    "distance": distance,
                    "similarity": 1.0 - distance,
                    "document": document
                }
            }

            candidates.append(candidate)

            logger.debug(f"Candidate {i+1}: {candidate['selector']} " +
                       f"(conf: {confidence:.3f}, dist: {distance:.3f}, " +
                       f"module: {metadata.get('module', 'unknown')})")

        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        # Get best candidate
        best_candidate = candidates[0]

        logger.debug("-" * 80)
        logger.debug(f"Best match: {best_candidate['selector']}")
        logger.debug(f"Confidence: {best_candidate['confidence']:.3f}")
        logger.debug(f"Agent: {best_candidate['agent_used']}")
        logger.debug("=" * 80)

        return {
            "selector_result": best_candidate,
            "candidates": candidates
        }


def test_agent1():
    """Test Agent 1 with sample queries"""

    print("=" * 80)
    print("Testing Agent 1: Selector Discovery")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Initialize Agent 1
    agent = Agent1SelectorDiscovery(config)

    # Test queries
    test_cases = [
        {
            "step_text": "Click the Save button",
            "module": "Teststep"
        },
        {
            "step_text": "Enter part name in the text field",
            "module": "CreateNew"
        },
        {
            "step_text": "Select type from dropdown",
            "module": "CreateNew"
        },
        {
            "step_text": "Navigate to Runs page",
            "module": "Common"
        },
        {
            "step_text": "Click edit icon for the run",
            "module": "Teststep"
        }
    ]

    print("\n")
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}")
        print("-" * 80)
        print(f"Step: {test['step_text']}")
        print(f"Module: {test['module']}")
        print("-" * 80)

        result = agent.discover_selector(
            step_text=test['step_text'],
            current_module=test['module']
        )

        if result['selector_result']:
            selector_result = result['selector_result']
            print(f"\n[OK] Found selector: {selector_result['selector']}")
            print(f"     Confidence: {selector_result['confidence']:.3f}")
            print(f"     Agent: {selector_result['agent_used']}")
            print(f"     Module: {selector_result['metadata']['module']}")
            print(f"     Priority: {selector_result['metadata']['priority']}")

            # Show top 3 candidates
            if len(result['candidates']) > 1:
                print(f"\n     Top candidates:")
                for j, candidate in enumerate(result['candidates'][:3], 1):
                    print(f"       {j}. {candidate['selector']} (conf: {candidate['confidence']:.3f})")
        else:
            print("[WARNING] No selector found")

        print("\n")

    print("=" * 80)
    print("Agent 1 testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_agent1()
    except Exception as e:
        print(f"\n[ERROR] Agent 1 test failed: {e}")
        logger.error(f"Agent 1 test failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
