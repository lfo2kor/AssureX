"""
PLCD Testing Assistant - Learning System
Saves successful runtime selectors to ChromaDB for future reuse
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from config_loader import load_config, get_azure_client, get_embedding_model, get_chroma_client


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/learning_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LearningSystem:
    """
    Learns from successful test executions and stores selectors in runtime collection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Learning System

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.azure_client = get_azure_client(config)
        self.embedding_model = get_embedding_model(config)
        self.chroma_client = get_chroma_client(config)

        # Get or create runtime collection
        self.runtime_collection_name = config['vector_database']['collections']['runtime_learned']
        self.runtime_collection = self.chroma_client.get_or_create_collection(
            name=self.runtime_collection_name
        )

        logger.info(f"Learning System initialized with collection: {self.runtime_collection_name}")


    def learn_from_execution(
        self,
        ticket_id: str,
        ticket_data: Dict,
        step_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Learn from successful test execution and store selectors

        Args:
            ticket_id: Jira ticket ID
            ticket_data: Parsed ticket data
            step_results: List of step execution results

        Returns:
            Dictionary with learning statistics
        """
        logger.info("=" * 80)
        logger.info("Learning System: Processing execution results")
        logger.info(f"Ticket: {ticket_id}")

        # Filter successful steps only
        successful_steps = [
            s for s in step_results
            if s.get('status') == 'PASSED' and s.get('selector')
        ]

        if not successful_steps:
            logger.info("No successful steps to learn from")
            return {"selectors_learned": 0, "successful_steps": 0}

        logger.info(f"Found {len(successful_steps)} successful steps to learn from")

        # Extract module
        module = ticket_data.get('module', 'Unknown')

        # Prepare selectors to add
        selectors_to_add = []
        embeddings_to_generate = []

        for step_result in successful_steps:
            step_num = step_result.get('step_number', 0)
            step_text = step_result.get('step_text', '')
            selector = step_result.get('selector', '')
            confidence = step_result.get('confidence', 0.0)
            agent_used = step_result.get('agent_used', 'Unknown')

            # Create selector metadata
            metadata = {
                'module': module,
                'selector': selector,
                'step_text': step_text,
                'ticket_id': ticket_id,
                'confidence': float(confidence),
                'agent_used': agent_used,
                'learned_at': datetime.now().isoformat(),
                'step_number': int(step_num)
            }

            # Create composite text for embedding (same strategy as Agent 1)
            composite_text = f"{step_text} {selector}"

            selectors_to_add.append({
                'id': f"{ticket_id}_step_{step_num}_{datetime.now().timestamp()}",
                'metadata': metadata,
                'text': composite_text
            })

            embeddings_to_generate.append(composite_text)

        # Generate embeddings in batch
        logger.info(f"Generating embeddings for {len(embeddings_to_generate)} selectors...")
        response = self.azure_client.embeddings.create(
            input=embeddings_to_generate,
            model=self.embedding_model
        )
        embeddings = [data.embedding for data in response.data]

        # Add to runtime collection
        ids = [s['id'] for s in selectors_to_add]
        metadatas = [s['metadata'] for s in selectors_to_add]
        documents = [s['text'] for s in selectors_to_add]

        logger.info(f"Adding {len(ids)} selectors to runtime collection...")
        self.runtime_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        logger.info(f"[OK] Learned {len(ids)} selectors from execution")
        logger.info("=" * 80)

        return {
            "selectors_learned": len(ids),
            "successful_steps": len(successful_steps),
            "collection": self.runtime_collection_name
        }


    def get_learned_selector(
        self,
        step_text: str,
        module: str,
        top_k: int = 3
    ) -> Optional[Dict]:
        """
        Query learned selectors from runtime collection

        Args:
            step_text: Test step description
            module: Target module
            top_k: Number of results to retrieve

        Returns:
            Best matching selector or None
        """
        # Generate embedding for step
        response = self.azure_client.embeddings.create(
            input=step_text,
            model=self.embedding_model
        )
        step_embedding = response.data[0].embedding

        # Query runtime collection
        results = self.runtime_collection.query(
            query_embeddings=[step_embedding],
            n_results=top_k,
            where={"module": module}
        )

        if not results['ids'] or len(results['ids'][0]) == 0:
            return None

        # Get best match
        best_metadata = results['metadatas'][0][0]
        best_distance = results['distances'][0][0]

        # Convert distance to similarity (ChromaDB uses L2 distance)
        # For cosine similarity approximation: similarity ≈ 1 - (distance² / 2)
        similarity = 1 - (best_distance ** 2) / 2

        return {
            'selector': best_metadata['selector'],
            'confidence': similarity,
            'source': 'runtime_learned',
            'original_step': best_metadata['step_text'],
            'original_ticket': best_metadata['ticket_id'],
            'learned_at': best_metadata['learned_at']
        }


    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learned selectors

        Returns:
            Dictionary with collection statistics
        """
        count = self.runtime_collection.count()

        # Get all items to analyze
        all_items = self.runtime_collection.get()

        modules = {}
        agents_used = {}

        if all_items['metadatas']:
            for metadata in all_items['metadatas']:
                module = metadata.get('module', 'Unknown')
                agent = metadata.get('agent_used', 'Unknown')

                modules[module] = modules.get(module, 0) + 1
                agents_used[agent] = agents_used.get(agent, 0) + 1

        return {
            'total_selectors': count,
            'modules': modules,
            'agents_used': agents_used,
            'collection_name': self.runtime_collection_name
        }


def test_learning_system():
    """Test the learning system"""
    print("=" * 80)
    print("Testing Learning System")
    print("=" * 80)

    # Load config
    config = load_config()

    # Initialize learning system
    learning_system = LearningSystem(config)

    # Get stats
    stats = learning_system.get_collection_stats()

    print(f"\n[OK] Learning System Stats:")
    print(f"     Total Selectors: {stats['total_selectors']}")
    print(f"     Modules: {stats['modules']}")
    print(f"     Agents Used: {stats['agents_used']}")
    print(f"     Collection: {stats['collection_name']}")

    print("\n" + "=" * 80)
    print("[OK] Learning System test complete")
    print("=" * 80)


if __name__ == "__main__":
    test_learning_system()
