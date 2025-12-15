"""
Insight Storage - Save and manage learnings from feedback system.
Handles storage in raw/, sessions/, consolidated/, and integrated/ folders.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class InsightStorage:
    """
    Manages storage and retrieval of insights from the feedback system.
    """

    def __init__(self, base_path: str = None):
        """
        Initialize insight storage.

        Args:
            base_path: Base directory for insights (default: ./insights)
        """
        if base_path is None:
            base_path = Path(__file__).parent / 'insights'
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.raw_path = base_path / 'raw'
        self.sessions_path = base_path / 'sessions'
        self.consolidated_path = base_path / 'consolidated'
        self.integrated_path = base_path / 'integrated'
        self.pending_path = base_path / 'pending'  # Pending insights (not yet embedded)

        # New feedback category paths
        self.corrections_path = base_path / 'corrections'  # False positives
        self.verified_path = base_path / 'verified'  # True positives (golden examples)
        self.negative_tests_path = base_path / 'negative_tests'  # Expected failures
        self.flaky_path = base_path / 'flaky'  # Timing issues
        self.bugs_path = base_path / 'bugs'  # Application bugs

        # Ensure directories exist
        for path in [self.raw_path, self.sessions_path, self.consolidated_path, self.integrated_path,
                     self.pending_path, self.corrections_path, self.verified_path,
                     self.negative_tests_path, self.flaky_path, self.bugs_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding client (lazy load)
        self._azure_client = None
        self._embedding_model = None

    def save_raw_insight(
        self,
        insight: Dict[str, Any],
        ticket_id: str,
        step_number: int,
        timestamp: str = None
    ) -> str:
        """
        Save a raw insight from a single failure.

        Args:
            insight: Complete insight dictionary
            ticket_id: Ticket identifier
            step_number: Failed step number
            timestamp: Optional timestamp (generated if not provided)

        Returns:
            Path to saved insight file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{ticket_id}_step{step_number}_{timestamp}.json"
        filepath = self.raw_path / filename

        # Add storage metadata
        storage_metadata = {
            'saved_at': datetime.now().isoformat(),
            'file_path': str(filepath),
            'storage_type': 'raw'
        }

        insight_with_metadata = {
            **insight,
            'storage_metadata': storage_metadata
        }

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insight_with_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n Raw insight saved: {filename}")

        return str(filepath)

    def save_session_data(
        self,
        session_data: Dict[str, Any],
        ticket_id: str,
        timestamp: str = None
    ) -> str:
        """
        Save session data for a complete test run.

        Args:
            session_data: Complete session information
            ticket_id: Ticket identifier
            timestamp: Optional timestamp

        Returns:
            Path to saved session file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{ticket_id}_{timestamp}_session.json"
        filepath = self.sessions_path / filename

        session_data['saved_at'] = datetime.now().isoformat()
        session_data['file_path'] = str(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"\n Session data saved: {filename}")

        return str(filepath)

    def load_raw_insight(self, filepath: str) -> Dict[str, Any]:
        """
        Load a raw insight from file.

        Args:
            filepath: Path to insight file

        Returns:
            Insight dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_raw_insights_for_ticket(self, ticket_id: str) -> List[Dict[str, Any]]:
        """
        Get all raw insights for a specific ticket.

        Args:
            ticket_id: Ticket identifier

        Returns:
            List of insight dictionaries
        """
        insights = []

        for filepath in self.raw_path.glob(f"{ticket_id}_step*.json"):
            try:
                insight = self.load_raw_insight(filepath)
                insights.append(insight)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return insights

    def get_all_raw_insights(self) -> List[Dict[str, Any]]:
        """
        Get all raw insights.

        Returns:
            List of all raw insights
        """
        insights = []

        for filepath in self.raw_path.glob("*.json"):
            try:
                insight = self.load_raw_insight(filepath)
                insights.append(insight)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

        return insights

    def consolidate_ticket_insights(self, ticket_id: str) -> Optional[str]:
        """
        Consolidate all raw insights for a ticket into one file.

        Args:
            ticket_id: Ticket identifier

        Returns:
            Path to consolidated file or None if no insights found
        """
        raw_insights = self.get_raw_insights_for_ticket(ticket_id)

        if not raw_insights:
            print(f"No raw insights found for {ticket_id}")
            return None

        # Create consolidated insight
        consolidated = {
            'ticket_id': ticket_id,
            'consolidated_at': datetime.now().isoformat(),
            'num_failures': len(raw_insights),
            'insights': raw_insights,
            'summary': self._create_summary(raw_insights)
        }

        # Save to consolidated folder
        filename = f"{ticket_id}_consolidated.json"
        filepath = self.consolidated_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        print(f"\n Consolidated {len(raw_insights)} insights for {ticket_id}")
        print(f"   Saved to: {filename}")

        return str(filepath)

    def _create_summary(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of multiple insights."""
        summary = {
            'total_insights': len(insights),
            'root_causes': [],
            'patterns_found': [],
            'average_confidence': 0.0,
            'reusability_scores': []
        }

        total_confidence = 0

        for insight in insights:
            # Extract root cause
            root_cause = insight.get('failure_analysis', {}).get('root_cause')
            if root_cause and root_cause not in summary['root_causes']:
                summary['root_causes'].append(root_cause)

            # Extract pattern
            pattern_name = insight.get('generalized_pattern', {}).get('pattern_name')
            if pattern_name and pattern_name not in summary['patterns_found']:
                summary['patterns_found'].append(pattern_name)

            # Collect confidence
            confidence = insight.get('learning_metadata', {}).get('confidence', 0)
            total_confidence += confidence

            # Collect reusability
            reusability = insight.get('learning_metadata', {}).get('reusability')
            if reusability:
                summary['reusability_scores'].append(reusability)

        # Calculate average confidence
        if len(insights) > 0:
            summary['average_confidence'] = total_confidence / len(insights)

        return summary

    def mark_as_integrated(self, consolidated_filepath: str) -> str:
        """
        Mark a consolidated insight as integrated into vector DB.

        Args:
            consolidated_filepath: Path to consolidated insight file

        Returns:
            Path to integrated file
        """
        # Load consolidated insight
        with open(consolidated_filepath, 'r', encoding='utf-8') as f:
            consolidated = json.load(f)

        # Add integration metadata
        consolidated['integrated_at'] = datetime.now().isoformat()
        consolidated['status'] = 'integrated'

        # Move to integrated folder
        filename = Path(consolidated_filepath).name.replace('_consolidated', '_integrated')
        filepath = self.integrated_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        print(f" Marked as integrated: {filename}")

        return str(filepath)

    def get_pending_insights(self) -> List[str]:
        """
        Get list of consolidated insights pending integration.

        Returns:
            List of file paths
        """
        pending = []

        for filepath in self.consolidated_path.glob("*.json"):
            # Check if already integrated
            integrated_filename = filepath.name.replace('_consolidated', '_integrated')
            integrated_path = self.integrated_path / integrated_filename

            if not integrated_path.exists():
                pending.append(str(filepath))

        return pending

    def get_insights_by_pattern(self, pattern_name: str) -> List[Dict[str, Any]]:
        """
        Find all insights that use a specific pattern.

        Args:
            pattern_name: Pattern name to search for

        Returns:
            List of matching insights
        """
        matching = []

        for filepath in self.raw_path.glob("*.json"):
            try:
                insight = self.load_raw_insight(filepath)
                insight_pattern = insight.get('generalized_pattern', {}).get('pattern_name', '')

                if pattern_name.lower() in insight_pattern.lower():
                    matching.append(insight)
            except Exception as e:
                continue

        return matching

    def _get_embedding_client(self):
        """Lazy load Azure OpenAI client for embedding generation."""
        if self._azure_client is None:
            from config_loader import get_azure_client, get_embedding_model, load_config
            config = load_config()
            self._azure_client = get_azure_client(config)
            self._embedding_model = get_embedding_model(config)
        return self._azure_client, self._embedding_model

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Azure OpenAI (same as Agent1).

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions for text-embedding-3-small)
        """
        try:
            client, model = self._get_embedding_client()
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Uses same formula as ChromaDB: similarity = 1 - cosine_distance

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)
        """
        if not vec1 or not vec2:
            return 0.0

        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate cosine similarity: dot(v1, v2) / (norm(v1) * norm(v2))
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            # Cosine similarity ranges from -1 to 1
            # Convert to 0 to 1 range: (similarity + 1) / 2
            cosine_sim = dot_product / (norm_v1 * norm_v2)

            # Return normalized similarity (same scale as ChromaDB)
            return float(cosine_sim)

        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def _detect_dynamic_selector(self, step_text: str, selector: str) -> tuple:
        """
        Detect if a selector is dynamic and extract the template value.

        Patterns detected:
        - "Choose Type 3" with [data-autocompleteitem="Type 3"] → dynamic, value="Type 3"
        - "Edit default_object_02" with [data-editBtn='default_object_02'] → dynamic, value="default_object_02"

        Args:
            step_text: The step description
            selector: The CSS selector

        Returns:
            Tuple of (is_dynamic: bool, template_value: str or None)
        """
        import re

        # Pattern 1: Extract values like "Type 3", "Type 1", "Measurement10", etc.
        # Matches: Capitalized word + space + number OR word + number
        dynamic_patterns = [
            r'\b([A-Z][a-z]+\s+\d+)\b',  # "Type 3", "User 5"
            r'\b([a-z_]+[a-z0-9_]*\d+[a-z0-9_]*)\b',  # "default_testobject_02", "item_123"
            r'\b([A-Z][a-z]+\d+)\b',  # "Measurement10", "Test5"
        ]

        for pattern in dynamic_patterns:
            # Find value in step text
            step_match = re.search(pattern, step_text)
            if step_match:
                value = step_match.group(1)
                # Check if this value appears in the selector
                if value in selector:
                    logger.debug(f"Dynamic selector detected: value='{value}' in selector='{selector}'")
                    return (True, value)

        return (False, None)

    def save_pending_insight(
        self,
        insight: Dict[str, Any],
        ticket_id: str,
        step_number: int,
        sequential_context: Dict[str, Any] = None,
        l1_attempts: List[Dict[str, Any]] = None,
        l2_attempts: List[Dict[str, Any]] = None,
        l3_attempts: List[Dict[str, Any]] = None,
        timestamp: str = None
    ) -> str:
        """
        Save pending insight with sequential context (not yet embedded to ChromaDB).

        Args:
            insight: Core insight data (selector, step_text, etc.)
            ticket_id: Ticket identifier
            step_number: Failed step number
            sequential_context: Previous steps and page state
            l1_attempts: List of L1 selector attempts
            l2_attempts: List of L2 DOM discovery attempts
            l3_attempts: List of L3 vision attempts
            timestamp: Optional timestamp

        Returns:
            Path to saved pending insight file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{ticket_id}_step{step_number}_{timestamp}.json"
        filepath = self.pending_path / filename

        # Extract module and action_type from sequential_context
        # FIX: Get module from insight directly (passed from results), not from sequential_context
        module = insight.get('module', '')  # Get from insight data (set by caller)
        if not module and sequential_context:
            module = sequential_context.get('current_module', '')  # Fallback to sequential_context
        action_type = ''
        if True:  # Keep original if sequential_context: block logic
            # Infer action type from step_text
            step_lower = insight.get('step_text', '').lower()
            if any(word in step_lower for word in ['edit', 'modify', 'update']):
                action_type = 'edit'
            elif any(word in step_lower for word in ['delete', 'remove']):
                action_type = 'delete'
            elif any(word in step_lower for word in ['save', 'submit', 'confirm']):
                action_type = 'save'
            elif any(word in step_lower for word in ['click', 'select', 'choose']):
                action_type = 'navigate'
            elif any(word in step_lower for word in ['verify', 'check', 'assert']):
                action_type = 'verify'
            else:
                action_type = 'interact'

        # Detect if this is a dynamic selector
        step_text = insight.get('step_text', '')
        selector = insight.get('correct_selector', '')
        is_dynamic, template_value = self._detect_dynamic_selector(step_text, selector)

        # Build metadata dictionary
        metadata = {
            'ticket_id': ticket_id,
            'step_number': step_number,
            'timestamp': datetime.now().isoformat()
        }

        # Add dynamic selector fields if detected
        if is_dynamic and template_value:
            metadata['isDynamic'] = True
            metadata['value'] = template_value
            logger.info(f"Dynamic selector auto-detected: value='{template_value}'")

        # Build complete pending insight with all required fields for batch embedding
        pending_insight = {
            'step': step_text,  # Required for embedding
            'selector': selector,  # Required for embedding
            'confidence': insight.get('confidence', 0.95),  # Default high confidence for tester feedback
            'category': insight.get('feedback_type', 'corrections'),  # corrections, verified, etc.

            'context': {  # Required for embedding
                'module': module,
                'action_type': action_type,
                'sequential_context': sequential_context or {}
            },

            'failure_analysis': {
                'l1_attempts': l1_attempts or [],
                'l2_attempts': l2_attempts or [],
                'l3_attempts': l3_attempts or [],
                'total_attempts': len(l1_attempts or []) + len(l2_attempts or []) + len(l3_attempts or []),
                'error_message': insight.get('error_message', '')
            },

            'metadata': metadata,  # Now includes isDynamic and value if detected

            'storage_metadata': {
                'saved_at': datetime.now().isoformat(),
                'file_path': str(filepath),
                'storage_type': 'pending',
                'embedded': False
            }
        }

        # Save to pending folder
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(pending_insight, f, indent=2, ensure_ascii=False)

        print(f"\n Pending insight saved: {filename}")

        return str(filepath)

    def search_pending_insights(
        self,
        step_text: str,
        sequential_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search pending insights for matching step using embedding-based semantic similarity.

        Args:
            step_text: Step description to match
            sequential_context: Optional previous step context for better matching

        Returns:
            Matching insight or None
        """
        best_match = None
        best_score = 0
        best_timestamp = ''

        logger.debug(f"Searching ALL insight categories with embedding-based similarity...")
        logger.debug(f"Query: '{step_text}'")

        # Generate embedding for query step
        query_embedding = self._generate_embedding(step_text)
        if not query_embedding:
            logger.warning("Failed to generate query embedding, falling back to string matching")
            return self._fallback_string_search(step_text, sequential_context)

        logger.debug(f"Generated query embedding ({len(query_embedding)} dimensions)")

        # Search ALL insight folders, not just pending
        search_paths = [
            self.corrections_path,   # False positives (highest priority)
            self.verified_path,      # True positives
            self.pending_path,       # Regular failures
            self.flaky_path,         # Timing issues
            self.negative_tests_path,# Expected failures
            self.bugs_path          # Application bugs
        ]

        all_files = []
        for path in search_paths:
            all_files.extend(list(path.glob("*.json")))

        logger.debug(f"Found {len(all_files)} total insight files across all categories")

        for filepath in all_files:
            try:
                logger.debug(f"Checking file: {filepath.name}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    insight = json.load(f)

                # Get stored embedding
                insight_embedding = insight.get('embedding', [])
                if not insight_embedding:
                    logger.debug(f"  No embedding found, skipping")
                    continue

                insight_step_text = insight.get('step', insight.get('step_text', ''))
                logger.debug(f"  Insight step: '{insight_step_text}'")

                # Calculate semantic similarity using embeddings
                semantic_similarity = self._cosine_similarity(query_embedding, insight_embedding)
                logger.debug(f"  Semantic similarity: {semantic_similarity:.3f}")

                # Start with semantic similarity as base score
                score = semantic_similarity

                # Boost score if sequential context matches
                context_boost = 0
                if sequential_context and insight.get('context', {}).get('sequential_context'):
                    last_action = sequential_context.get('last_successful_action') or ''
                    insight_context = insight['context']['sequential_context']
                    insight_last_action = insight_context.get('last_successful_action') or ''

                    last_action = last_action.lower() if last_action else ''
                    insight_last_action = insight_last_action.lower() if insight_last_action else ''

                    if last_action and insight_last_action and last_action in insight_last_action:
                        context_boost = 0.05  # Small boost for sequential context
                        score += context_boost
                        logger.debug(f"  Sequential context match! Boost: +{context_boost:.2f}, New score: {score:.3f}")

                logger.debug(f"  Final score: {score:.3f}, Best score so far: {best_score:.3f}")

                # Get timestamp for tie-breaking
                current_ts = insight.get('metadata', {}).get('timestamp', '')
                if not current_ts:
                    current_ts = insight.get('storage_metadata', {}).get('saved_at', '')

                # Update if score is better, or if similar score but newer timestamp
                should_update = False
                if score > best_score and score > 0.75:  # Threshold: 0.75 semantic similarity
                    should_update = True
                    logger.debug(f"  Better score!")
                elif abs(score - best_score) < 0.02 and best_match:  # Very close scores
                    # Tie-breaker: Use most recent timestamp
                    if current_ts > best_timestamp:
                        should_update = True
                        logger.debug(f"  Similar score but newer timestamp: {current_ts} > {best_timestamp}")

                if should_update:
                    best_score = score
                    best_match = insight
                    best_timestamp = current_ts
                    logger.debug(f"  New best match! Score: {best_score:.3f}, File: {filepath.name}")
                else:
                    logger.debug(f"  Not better than best_score ({best_score:.3f})")

            except Exception as e:
                logger.debug(f"  Error processing file: {e}")
                continue

        logger.debug(f"Final result: best_match={'Yes' if best_match else 'No'}, best_score={best_score:.3f}")

        if best_match:
            logger.info(f"✓ Found pending insight match (semantic similarity: {best_score:.3f})")

        return best_match

    def _fallback_string_search(
        self,
        step_text: str,
        sequential_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback to simple string matching if embedding generation fails.

        Args:
            step_text: Step description to match
            sequential_context: Optional previous step context

        Returns:
            Matching insight or None
        """
        best_match = None
        best_score = 0

        search_paths = [
            self.corrections_path,
            self.verified_path,
            self.pending_path,
            self.flaky_path,
            self.negative_tests_path,
            self.bugs_path
        ]

        all_files = []
        for path in search_paths:
            all_files.extend(list(path.glob("*.json")))

        for filepath in all_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    insight = json.load(f)

                insight_step_text = insight.get('step', insight.get('step_text', '')).lower()
                query_step_text = step_text.lower()

                score = 0
                if query_step_text in insight_step_text or insight_step_text in query_step_text:
                    score = 0.8

                if sequential_context and insight.get('context', {}).get('sequential_context'):
                    last_action = sequential_context.get('last_successful_action') or ''
                    insight_last_action = insight['context']['sequential_context'].get('last_successful_action') or ''

                    if last_action and insight_last_action and last_action.lower() in insight_last_action.lower():
                        score += 0.2

                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = insight

            except Exception as e:
                continue

        return best_match

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored insights.

        Returns:
            Dictionary with statistics
        """
        raw_count = len(list(self.raw_path.glob("*.json")))
        session_count = len(list(self.sessions_path.glob("*.json")))
        consolidated_count = len(list(self.consolidated_path.glob("*.json")))
        integrated_count = len(list(self.integrated_path.glob("*.json")))
        pending_count = len(list(self.pending_path.glob("*.json")))  # Updated
        pending_integration_count = len(self.get_pending_insights())

        # Get unique tickets
        tickets = set()
        for filepath in self.raw_path.glob("*.json"):
            ticket_id = filepath.name.split('_step')[0]
            tickets.add(ticket_id)

        return {
            'total_raw_insights': raw_count,
            'total_sessions': session_count,
            'total_consolidated': consolidated_count,
            'total_integrated': integrated_count,
            'pending_insights': pending_count,  # New
            'pending_integration': pending_integration_count,
            'unique_tickets': len(tickets),
            'ticket_ids': sorted(list(tickets))
        }

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print(" INSIGHT STORAGE STATISTICS")
        print("="*60)
        print(f"Raw insights collected: {stats['total_raw_insights']}")
        print(f"Test sessions: {stats['total_sessions']}")
        print(f"Pending insights (not embedded): {stats['pending_insights']}")
        print(f"Consolidated insights: {stats['total_consolidated']}")
        print(f"Integrated to vector DB: {stats['total_integrated']}")
        print(f"Pending integration: {stats['pending_integration']}")
        print(f"Unique tickets: {stats['unique_tickets']}")
        if stats['ticket_ids']:
            print(f"Tickets: {', '.join(stats['ticket_ids'][:10])}")
            if len(stats['ticket_ids']) > 10:
                print(f"         ... and {len(stats['ticket_ids']) - 10} more")
        print("="*60 + "\n")

    def save_insight_by_category(
        self,
        insight: Dict[str, Any],
        ticket_id: str,
        step_number: int
    ) -> str:
        """
        Save insight to appropriate category folder based on feedback_type.

        SAFETY: This method ONLY writes to JSON files in insights/ folders.
        It does NOT interact with ChromaDB vector database.
        ChromaDB updates happen ONLY during batch embedding (setup_vectordb.py).

        Args:
            insight: Enhanced insight dictionary with feedback_type
            ticket_id: Ticket identifier
            step_number: Step number

        Returns:
            Path to saved file
        """
        feedback_type = insight.get('feedback_type', 'failed')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine target path based on feedback type
        category_map = {
            'false_positive': self.corrections_path,
            'failed': self.pending_path,
            'timing_issue': self.flaky_path,
            'true_positive': self.verified_path,
            'application_bug': self.bugs_path,
            'negative_test': self.negative_tests_path
        }

        target_path = category_map.get(feedback_type, self.pending_path)

        # Generate filename
        filename = f"{ticket_id}_step{step_number}_{timestamp}.json"
        filepath = target_path / filename

        # Add storage metadata
        insight_with_metadata = {
            **insight,
            'storage_metadata': {
                'saved_at': datetime.now().isoformat(),
                'file_path': str(filepath),
                'storage_type': feedback_type,
                'embedded': False
            }
        }

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insight_with_metadata, f, indent=2, ensure_ascii=False)

        print(f" Saved {feedback_type} insight: {filename}")

        return str(filepath)


# CLI for management
if __name__ == "__main__":
    import sys

    storage = InsightStorage()

    if len(sys.argv) < 2:
        print("Insight Storage Management")
        print("\nUsage:")
        print("  python insight_storage.py stats              - Show statistics")
        print("  python insight_storage.py consolidate <ticket>  - Consolidate ticket insights")
        print("  python insight_storage.py consolidate-all    - Consolidate all tickets")
        print("  python insight_storage.py pending            - List pending integrations")
        print("  python insight_storage.py pattern <name>     - Find insights by pattern")
        sys.exit(0)

    command = sys.argv[1]

    if command == "stats":
        storage.print_statistics()

    elif command == "consolidate":
        if len(sys.argv) < 3:
            print("Error: Please provide ticket ID")
            sys.exit(1)

        ticket_id = sys.argv[2]
        result = storage.consolidate_ticket_insights(ticket_id)

        if result:
            print(f" Consolidation complete: {result}")
        else:
            print(f" No insights found for {ticket_id}")

    elif command == "consolidate-all":
        stats = storage.get_statistics()
        tickets = stats['ticket_ids']

        print(f"Consolidating {len(tickets)} tickets...")

        for ticket_id in tickets:
            storage.consolidate_ticket_insights(ticket_id)

        print(f"\n Consolidated all tickets")

    elif command == "pending":
        pending = storage.get_pending_insights()

        print(f"\nPending integration: {len(pending)} insights")
        for filepath in pending:
            print(f"  - {Path(filepath).name}")

    elif command == "pattern":
        if len(sys.argv) < 3:
            print("Error: Please provide pattern name")
            sys.exit(1)

        pattern_name = sys.argv[2]
        insights = storage.get_insights_by_pattern(pattern_name)

        print(f"\nFound {len(insights)} insights matching '{pattern_name}':")
        for insight in insights:
            ticket = insight.get('metadata', {}).get('ticket_id', 'Unknown')
            step = insight.get('metadata', {}).get('failed_step', '?')
            pattern = insight.get('generalized_pattern', {}).get('pattern_name', 'N/A')
            print(f"  - {ticket} (step {step}): {pattern}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
