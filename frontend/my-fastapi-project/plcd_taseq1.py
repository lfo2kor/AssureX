"""
PLCD Testing Assistant - Sequential Context Tracking Version
Uses LangGraph multi-agent architecture for intelligent test execution
"""

import sys
import os
# sys.path.append(r"C:\Idea Projects\PLCD_TA_Team\PLCD_TA_Team")
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime

from playwright.sync_api import sync_playwright, Page, Browser

from config_loader import load_config, get_azure_client
from report_generator import generate_html_report
from script_generator import generate_playwright_script

from dotenv import load_dotenv
load_dotenv()
external_project_path = os.getenv("EXTERNAL_PROJECT_PATH", None)

if external_project_path:
    sys.path.append(external_project_path)
else:
    print("[WARNING] EXTERNAL_PROJECT_PATH not set. Using local project only.")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/plcd_taseq.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# TYPE DEFINITIONS - LangGraph State
# ============================================================================

class TestExecutionState(TypedDict):
    """Shared state across all agents in LangGraph workflow"""

    # Test metadata
    ticket_id: str
    ticket_title: str
    module: str
    current_step: int
    total_steps: int

    # Context tracking
    context_history: List[Dict[str, Any]]  # Historical context per step
    current_context: Dict[str, Any]         # Current page context

    # Selector discovery
    selector_attempts: List[Dict[str, Any]]  # All attempts
    successful_selectors: Dict[str, str]     # step_text -> selector

    # Learning
    failure_patterns: List[Dict[str, Any]]
    corrections_used: List[Dict[str, Any]]

    # Agent routing
    agent_chain: List[str]                   # Which agents were called
    next_agent: str                          # Next agent to call
    orchestrator_reasoning: List[str]        # Decision log

    # Execution results
    step_results: List[Dict[str, Any]]
    overall_status: str

    # Artifacts
    video_path: Optional[str]
    script_path: Optional[str]
    report_path: Optional[str]

    # Runtime data (not serialized to JSON)
    execution_start_time: float
    config: Dict[str, Any]


# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Base class for all agents"""

    def __init__(self, config: Dict[str, Any], agent_name: str):
        """
        Initialize base agent

        Args:
            config: Configuration dictionary
            agent_name: Name of the agent (e.g., 'orchestrator_agent')
        """
        self.config = config
        self.agent_name = agent_name
        self.agent_config = config.get('agents', {}).get(agent_name, {})
        self.azure_client = get_azure_client(config)

        # Agent settings
        self.enabled = self.agent_config.get('enabled', True)
        self.model = self.agent_config.get('model', 'gpt-4o')
        self.temperature = self.agent_config.get('temperature', 0.1)
        self.max_tokens = self.agent_config.get('max_tokens', 1000)

        logger.info(f"{agent_name} initialized (model: {self.model})")

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Azure OpenAI LLM

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            LLM response text
        """
        try:
            response = self.azure_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{self.agent_name} LLM call failed: {e}")
            raise


# ============================================================================
# JIRA AGENT - LLM-Based Ticket Parsing
# ============================================================================

class JiraAgent(BaseAgent):
    """Parse Jira tickets using LLM (no regex)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'jira_agent')
        self.system_prompt = self.agent_config.get('system_prompt', '')
        self.format_examples = self.agent_config.get('format_examples', '')

    def parse_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """
        Parse Jira ticket using LLM

        Args:
            ticket_id: Jira ticket ID (e.g., "RBPLCD-8835")

        Returns:
            Parsed ticket data with steps
        """
        logger.info(f"JiraAgent: Parsing ticket {ticket_id}")

        # Read ticket file
        jira_folder = self.config['folders']['jira']
        ticket_path = Path(jira_folder) / f"{ticket_id}.txt"

        if not ticket_path.exists():
            raise FileNotFoundError(f"Ticket file not found: {ticket_path}")

        with open(ticket_path, 'r', encoding='utf-8') as f:
            ticket_content = f.read()

        # Build LLM prompt
        user_prompt = f"""
Ticket ID: {ticket_id}

Ticket Content:
{ticket_content}

Format Examples:
{self.format_examples}

Parse this ticket and extract test steps in JSON format.
"""

        # Call LLM
        response = self.call_llm(self.system_prompt, user_prompt)

        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            parsed_data = json.loads(json_str)

            logger.info(f"JiraAgent: Parsed {len(parsed_data.get('steps', []))} steps")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JiraAgent: Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response}")
            raise


# ============================================================================
# CONTEXT AGENT - Sequential Context Tracking
# ============================================================================

class ContextAgent(BaseAgent):
    """Track execution context throughout test run"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'context_agent')
        self.memory_config = config.get('memory', {}).get('context_agent', {})
        self.retention_steps = self.memory_config.get('retention_steps', 10)

    def capture_context(self, page: Page) -> Dict[str, Any]:
        """
        Capture current page context

        Args:
            page: Playwright page object

        Returns:
            Context dictionary
        """
        try:
            # Execute JavaScript to extract context
            context = page.evaluate("""
                () => ({
                    url: window.location.href,
                    pathname: window.location.pathname,
                    visible_elements: Array.from(document.querySelectorAll('*'))
                        .filter(el => el.offsetParent !== null)
                        .map(el => Array.from(el.attributes)
                            .filter(a => a.name.startsWith('data-'))
                            .map(a => `${a.name}='${a.value}'`))
                        .flat()
                        .filter(v => v.length > 0)
                        .slice(0, 100),  // Limit to 100
                    breadcrumb: Array.from(document.querySelectorAll('[data-breadcrumb] span, .breadcrumb span'))
                        .map(el => el.textContent.trim())
                        .filter(t => t.length > 0),
                    page_title: document.title,
                    timestamp: new Date().toISOString()
                })
            """)

            logger.info(f"ContextAgent: Captured context from {context['pathname']}")
            logger.info(f"ContextAgent: Found {len(context['visible_elements'])} visible data-* elements")

            return context

        except Exception as e:
            logger.error(f"ContextAgent: Failed to capture context: {e}")
            return {
                "url": page.url,
                "visible_elements": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def update_history(self, state: TestExecutionState, new_context: Dict[str, Any]):
        """
        Update context history with sliding window

        Args:
            state: Current execution state
            new_context: New context to add
        """
        # Add to history
        state['context_history'].append(new_context)

        # Keep only last N steps (sliding window)
        if len(state['context_history']) > self.retention_steps:
            state['context_history'] = state['context_history'][-self.retention_steps:]

        # Update current context
        state['current_context'] = new_context

    def generate_context_summary(self, state: TestExecutionState) -> str:
        """
        Generate human-readable context summary for other agents

        Args:
            state: Current execution state

        Returns:
            Context summary string
        """
        current = state['current_context']
        history = state['context_history']

        summary = f"""
Current Page Context:
- URL: {current.get('url', 'Unknown')}
- Module: {state['module']}
- Visible elements: {len(current.get('visible_elements', []))} data-* attributes
- Breadcrumb: {' > '.join(current.get('breadcrumb', []))}

Recent History ({len(history)} steps):
"""

        for i, ctx in enumerate(history[-3:], 1):  # Last 3 steps
            summary += f"{i}. {ctx.get('url', 'Unknown')}\n"

        return summary


# ============================================================================
# LEARNING AGENT - Continuous Learning from Failures/Corrections
# ============================================================================

class LearningAgent(BaseAgent):
    """Query and store learned selectors with context"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'learning_agent')

        # Import ChromaDB dependencies
        from config_loader import get_chroma_client, get_embedding_model

        self.chroma_client = get_chroma_client(config)
        self.embedding_model = get_embedding_model(config)

        # Get or create learning collection
        collection_name = config.get('vector_database', {}).get('collections', {}).get('runtime_learned', 'learning_collection')
        self.learning_collection = self.chroma_client.get_or_create_collection(name=collection_name)

        self.similarity_threshold = config.get('memory', {}).get('learning_agent', {}).get('similarity_threshold', 0.85)

        logger.info(f"LearningAgent: Collection '{collection_name}' initialized")

    def query_learned_selector(self, step_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query learning collection for similar selector with matching context

        Args:
            step_text: Step description
            context: Current page context

        Returns:
            Learned selector with confidence, or None
        """
        try:
            # Build contextual query
            url = context.get('url', '')
            visible = context.get('visible_elements', [])[:10]  # First 10

            query_text = f"{step_text} | URL: {url} | Visible: {', '.join(visible)}"

            # Generate embedding
            from openai import AzureOpenAI
            azure_client = self.azure_client

            response = azure_client.embeddings.create(
                input=query_text,
                model=self.config['azure_openai']['models']['embedding']
            )
            query_embedding = response.data[0].embedding

            # Query ChromaDB
            results = self.learning_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            if not results['ids'] or len(results['ids'][0]) == 0:
                logger.info("LearningAgent: No learned selectors found")
                return None

            # Get best match
            best_match = results['metadatas'][0][0] if results['metadatas'] else None
            distance = results['distances'][0][0] if results['distances'] else 1.0

            # Convert distance to confidence (cosine similarity)
            confidence = 1.0 - distance

            if confidence >= self.similarity_threshold and best_match:
                logger.info(f"LearningAgent: Found learned selector (conf: {confidence:.2f})")
                return {
                    'selector': best_match.get('selector'),
                    'confidence': confidence,
                    'source': best_match.get('source', 'learned'),
                    'verified': best_match.get('verified', False)
                }

            logger.info(f"LearningAgent: Best match confidence {confidence:.2f} below threshold {self.similarity_threshold}")
            return None

        except Exception as e:
            logger.error(f"LearningAgent: Query failed: {e}")
            return None

    def store_learned_selector(self, step_text: str, selector: str, context: Dict[str, Any],
                                confidence: float, source: str = 'runtime', verified: bool = True):
        """
        Store learned selector with context

        Args:
            step_text: Step description
            selector: Selector that worked
            context: Context where it worked
            confidence: Confidence score
            source: Source (runtime, human_feedback)
            verified: Whether verified
        """
        try:
            # Generate unique ID
            import hashlib
            id_str = f"{step_text}_{selector}_{context.get('url', '')}"
            doc_id = hashlib.md5(id_str.encode()).hexdigest()

            # Build document text
            url = context.get('url', '')
            visible = context.get('visible_elements', [])[:10]
            doc_text = f"{step_text} | URL: {url} | Selector: {selector} | Visible: {', '.join(visible)}"

            # Generate embedding
            response = self.azure_client.embeddings.create(
                input=doc_text,
                model=self.config['azure_openai']['models']['embedding']
            )
            embedding = response.data[0].embedding

            # Store to ChromaDB
            self.learning_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{
                    'step_text': step_text,
                    'selector': selector,
                    'url_pattern': url,
                    'confidence': confidence,
                    'source': source,
                    'verified': verified,
                    'timestamp': datetime.now().isoformat()
                }]
            )

            logger.info(f"LearningAgent: Stored selector for '{step_text}'")

        except Exception as e:
            logger.error(f"LearningAgent: Storage failed: {e}")


# ============================================================================
# SELECTOR AGENT L1 - RAG + LLM Validation
# ============================================================================

class SelectorAgentL1(BaseAgent):
    """RAG-based selector discovery with LLM validation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'selector_agent_l1')

        # Import existing Agent1 for RAG functionality
        from agent1_selector_discovery import Agent1SelectorDiscovery
        self.agent1 = Agent1SelectorDiscovery(config)

        self.confidence_threshold = self.agent_config.get('confidence_threshold', 0.75)
        self.retry_threshold = self.agent_config.get('retry_threshold', 0.70)

    def discover_selector(self, step_text: str, context: Dict[str, Any], state: TestExecutionState) -> Dict[str, Any]:
        """
        Discover selector using RAG + LLM validation

        Args:
            step_text: Step description
            context: Current page context
            state: Execution state

        Returns:
            Discovery result with selector and confidence
        """
        logger.info(f"SelectorAgent_L1: Discovering selector for '{step_text}'")

        # Step 1: LLM enhances query with context
        enhanced_query = self._enhance_query_with_context(step_text, context, state)

        # Extract page context for metadata filtering
        from urllib.parse import urlparse
        url = context.get('url', '')
        try:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p]
            page_context = parts[-1] if parts else ''
        except:
            page_context = ''

        # Skip page_context filter for navigation/sidebar and dropdown selection queries
        # (nav elements are visible on multiple pages, dropdown options are module-specific)
        nav_keywords = ['navigate', 'sidebar', 'nav item', 'navigation', 'nav-']
        dropdown_keywords = ['select', 'choose', 'pick', 'dropdown', 'menu']

        is_nav_query = any(keyword in step_text.lower() for keyword in nav_keywords)
        is_dropdown_query = any(kw in step_text.lower() for kw in dropdown_keywords)

        if is_nav_query:
            logger.info(f"Navigation query detected, skipping page_context filter")
            page_context = None
        elif is_dropdown_query:
            logger.info(f"Dropdown selection query detected, skipping page_context filter")
            page_context = None

        # Step 2: RAG search with enhanced query and metadata filtering
        # For dropdown queries, increase n_results to capture EntityAttribute selectors
        n_results = 15 if is_dropdown_query else None
        rag_result = self.agent1.discover_selector(
            step_text=enhanced_query,
            current_module=state['module'],
            page_context=page_context,
            n_results=n_results
        )

        if not rag_result or not rag_result.get('selector_result'):
            logger.info("SelectorAgent_L1: No RAG results found")
            return {'selector': None, 'confidence': 0.0, 'agent': 'L1', 'reasoning': 'No RAG match'}

        # Step 2.5: Re-rank candidates if step contains specific names
        # Generic: Extract names from step text and boost selectors containing those names
        import re
        # Find words that look like identifiers with underscores or mixed alphanumeric
        # Pattern: word_word, word_word_01, default_Measurement01, test_object, etc.
        name_patterns = re.findall(r'\b[a-zA-Z]+[_][a-zA-Z0-9_]+\b', step_text)

        if name_patterns and rag_result.get('candidates'):
            logger.info(f"Detected name patterns in step: {name_patterns}")
            # Check all candidates for name matches
            for candidate in rag_result['candidates']:
                selector = candidate.get('selector', '')
                # Check if selector contains any of the detected names
                for name in name_patterns:
                    if name.lower() in selector.lower():
                        logger.info(f"Name match found: '{name}' in {selector}, boosting confidence")
                        candidate['confidence'] = min(candidate['confidence'] + 0.20, 1.0)
                        break

            # Re-sort candidates by confidence
            rag_result['candidates'].sort(key=lambda x: x['confidence'], reverse=True)
            # Update best result
            rag_result['selector_result'] = rag_result['candidates'][0]

        # Step 2.6: Action keyword matching (edit vs delete, save vs cancel, etc.)
        # Boost selectors with matching action keywords, penalize conflicting ones
        action_keywords = {
            'edit': {'boost': ['edit', 'modify', 'update'], 'penalize': ['delete', 'remove', 'cancel']},
            'delete': {'boost': ['delete', 'remove'], 'penalize': ['edit', 'save', 'update']},
            'save': {'boost': ['save', 'submit', 'confirm'], 'penalize': ['cancel', 'delete']},
            'cancel': {'boost': ['cancel', 'close'], 'penalize': ['save', 'submit']},
        }

        step_lower = step_text.lower()
        for action, keywords in action_keywords.items():
            if action in step_lower and rag_result.get('candidates'):
                logger.info(f"Detected '{action}' action in step, applying keyword matching")
                for candidate in rag_result['candidates']:
                    selector_lower = candidate.get('selector', '').lower()
                    # Boost matching keywords
                    for boost_word in keywords['boost']:
                        if boost_word in selector_lower:
                            logger.info(f"Action match: '{boost_word}' in {candidate['selector']}, boosting")
                            candidate['confidence'] = min(candidate['confidence'] + 0.15, 1.0)
                            break
                    # Penalize conflicting keywords
                    for penalize_word in keywords['penalize']:
                        if penalize_word in selector_lower:
                            logger.info(f"Action conflict: '{penalize_word}' in {candidate['selector']}, penalizing")
                            candidate['confidence'] = max(candidate['confidence'] - 0.25, 0.40)
                            break

                # Re-sort after action keyword matching
                rag_result['candidates'].sort(key=lambda x: x['confidence'], reverse=True)
                rag_result['selector_result'] = rag_result['candidates'][0]
                break  # Only apply one action keyword set

        # Step 2.7: Sequential context - avoid reusing previous step's selector
        # If previous step clicked a dropdown/field, this step should select an option (not the same field)
        previous_selector = None
        if state.get('step_results') and len(state['step_results']) > 0:
            previous_result = state['step_results'][-1]
            previous_selector = previous_result.get('selector')
            previous_step_text = previous_result.get('text', '').lower()

            # Check if previous step was "click dropdown/field" and current is "select/choose from"
            prev_dropdown_keywords = ['dropdown', 'field', 'click on']
            curr_select_keywords = ['select', 'choose', 'pick']

            is_prev_dropdown = any(kw in previous_step_text for kw in prev_dropdown_keywords)
            is_curr_select = any(kw in step_lower for kw in curr_select_keywords)

            if is_prev_dropdown and is_curr_select and previous_selector and rag_result.get('candidates'):
                logger.info(f"Sequential context: Previous step clicked {previous_selector}, penalizing same selector")
                for candidate in rag_result['candidates']:
                    if candidate.get('selector') == previous_selector:
                        logger.info(f"Penalizing duplicate selector from previous step: {previous_selector}")
                        candidate['confidence'] = max(candidate['confidence'] - 0.40, 0.30)

        # Step 2.8: Dropdown selection value matching with dynamic selector support
        # For "Select X from dropdown" steps, handle dynamic selectors like [data-autocompleteitem="{{item}}"]
        select_keywords = ['select', 'choose', 'pick']
        dropdown_keywords = ['dropdown', 'menu', 'list']
        if any(kw in step_lower for kw in select_keywords) and any(kw in step_lower for kw in dropdown_keywords):
            # Extract the value being selected - words between select/choose and from/dropdown
            import re
            # Pattern: "select <value> from" or "select <value> dropdown"
            value_match = re.search(r'(?:select|choose|pick)\s+([^from]+?)\s+(?:from|in|dropdown)', step_lower)
            if value_match and rag_result.get('candidates'):
                value_to_select = value_match.group(1).strip()
                logger.info(f"Dropdown selection detected: '{value_to_select}'")

                # If no autoCompleteItem selector found in candidates, explicitly add it
                has_autocomplete = any('autocompleteitem' in c.get('selector', '').lower() for c in rag_result['candidates'])
                if not has_autocomplete:
                    logger.info("No autoCompleteItem selector in candidates, adding it manually")
                    # Create a dynamic autoCompleteItem candidate with the value
                    concrete_selector = f"[data-autoCompleteItem='{value_to_select.title()}']"
                    rag_result['candidates'].append({
                        'selector': concrete_selector,
                        'confidence': 0.70,  # High confidence for dropdown option
                        'agent_used': 'L1',
                        'metadata': {
                            'isDynamic': True,
                            'module': 'EntityAttribute',
                            'attr': 'data-autoCompleteItem'
                        }
                    })
                    logger.info(f"Added dynamic selector: {concrete_selector} with conf: 0.70")

                # Check for dynamic selectors with {{item}} or similar patterns
                dynamic_found = False
                for candidate in rag_result['candidates']:
                    selector = candidate.get('selector', '')
                    selector_lower = selector.lower()
                    metadata = candidate.get('metadata', {})
                    is_dynamic = metadata.get('isDynamic', False)

                    # Handle dynamic selectors: Replace {{item}} with actual value
                    if is_dynamic and ('{{item}}' in selector or '{{' in selector):
                        # Replace placeholder with actual value (case-preserving)
                        concrete_selector = selector.replace('{{item}}', value_to_select.title())
                        concrete_selector = concrete_selector.replace("'{{item}}'", f"'{value_to_select.title()}'")
                        concrete_selector = concrete_selector.replace('"{{item}}"', f'"{value_to_select.title()}"')

                        logger.info(f"Dynamic selector found: {selector} -> {concrete_selector}")
                        candidate['selector'] = concrete_selector
                        candidate['confidence'] = min(candidate['confidence'] + 0.35, 1.0)  # Big boost for dynamic match
                        dynamic_found = True
                    # Boost static selectors containing the value
                    elif value_to_select in selector_lower:
                        logger.info(f"Value match: '{value_to_select}' in {selector}, boosting")
                        candidate['confidence'] = min(candidate['confidence'] + 0.25, 1.0)
                    # Penalize field selectors (they open the dropdown, not select the value)
                    elif 'attribute' in selector_lower or 'field' in selector_lower:
                        logger.info(f"Field selector detected (not value): {selector}, penalizing")
                        candidate['confidence'] = max(candidate['confidence'] - 0.20, 0.40)

                # Boost option/item selectors (autocomplete, option, listitem, etc.)
                option_keywords = ['autocompleteitem', 'option', 'listitem', 'menuitem']
                for candidate in rag_result['candidates']:
                    selector_lower = candidate.get('selector', '').lower()
                    metadata = candidate.get('metadata', {})
                    is_dynamic = metadata.get('isDynamic', False)

                    if any(kw in selector_lower for kw in option_keywords):
                        logger.info(f"Option selector detected: {candidate['selector']}, boosting")
                        # Extra boost for dynamic selectors
                        boost = 0.30 if is_dynamic else 0.15
                        candidate['confidence'] = min(candidate['confidence'] + boost, 1.0)

                # Re-sort after dropdown value matching
                rag_result['candidates'].sort(key=lambda x: x['confidence'], reverse=True)
                rag_result['selector_result'] = rag_result['candidates'][0]

        selector_result = rag_result['selector_result']
        selector = selector_result['selector']
        base_confidence = selector_result['confidence']

        # Step 3: Validate against visible DOM
        validated_result = self._validate_with_llm(selector, base_confidence, step_text, context)

        logger.info(f"SelectorAgent_L1: Result - {validated_result['selector']} (conf: {validated_result['confidence']:.2f})")

        return validated_result

    def _enhance_query_with_context(self, step_text: str, context: Dict[str, Any], state: TestExecutionState) -> str:
        """
        Build natural language query that matches ChromaDB embedding format

        Instead of LLM enhancement, extract runtime context and build query
        in same natural language format as ChromaDB embeddings.
        """
        try:
            # Extract page context from URL
            from urllib.parse import urlparse
            url = context.get('url', '')
            try:
                path = urlparse(url).path
                parts = [p for p in path.split('/') if p]
                page_context = parts[-1] if parts else ''
            except:
                page_context = ''

            # Extract visible context hints from data attributes
            visible_elements = context.get('visible_elements', [])
            visible_context = self._extract_context_hints(visible_elements)

            # Get previous step for sequence
            context_history = state.get('context_history', [])
            previous_step = ''
            if context_history and len(context_history) > 0:
                prev = context_history[-1].get('step', '')
                if prev:
                    previous_step = f"after {prev}"

            # Build natural language query similar to ChromaDB format
            # ChromaDB has: "Click link to navigate to Teststep runs from sidebar on dashboard page"
            # Runtime builds: "Navigate to Teststep from sidebar on dashboard page"

            query_parts = [step_text]

            # Only add navigation context for actual navigation steps
            # Avoid adding "from sidebar" for form field interactions (dropdown, text field, etc.)
            step_lower = step_text.lower()
            is_navigation_step = any(nav_word in step_lower for nav_word in ['navigate', 'go to', 'open', 'link']) or \
                                 'sidebar' in step_lower or \
                                 'menu' in step_lower
            is_form_field_step = any(field_word in step_lower for field_word in ['dropdown', 'field', 'input', 'text', 'button', 'checkbox'])

            # Only add visible context for navigation steps, not form interactions
            if visible_context and is_navigation_step and not is_form_field_step:
                query_parts.append(f"from {' '.join(visible_context[:2])}")

            if page_context:
                query_parts.append(f"on {page_context} page")

            if previous_step:
                query_parts.append(previous_step)

            enhanced_query = ' '.join(query_parts)

            logger.info(f"SelectorAgent_L1: Enhanced query: '{enhanced_query}'")
            return enhanced_query

        except Exception as e:
            logger.warning(f"SelectorAgent_L1: Query building failed, using original: {e}")
            return step_text

    def _extract_context_hints(self, visible_elements: list) -> list:
        """
        Extract context keywords from visible data attributes

        Args:
            visible_elements: List of visible data attribute strings

        Returns:
            List of context keywords
        """
        hints = set()

        for elem in visible_elements:
            # Extract value from data attributes
            # "data-test='sidebar-nav-item-nav_item_teststeps'" → extract keywords
            if '=' in elem:
                try:
                    value = elem.split('=')[1].strip("'\"")
                    # Split by common separators
                    parts = value.replace('-', ' ').replace('_', ' ').split()
                    # Keep meaningful keywords
                    for part in parts:
                        if len(part) > 2 and part.lower() not in ['item', 'data', 'test', 'btn', 'id']:
                            hints.add(part.lower())
                except:
                    continue

        # Filter for navigation/UI keywords
        nav_keywords = ['sidebar', 'navigation', 'nav', 'menu', 'toolbar', 'header', 'footer', 'panel']
        found = [h for h in hints if h in nav_keywords]

        return found[:3]  # Return top 3

    def _validate_with_llm(self, selector: str, confidence: float, step_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RAG result against visible DOM - Simple string matching"""
        try:
            visible = context.get('visible_elements', [])
            logger.info(f"Validating selector {selector} against {len(visible)} visible elements on page {context.get('url', 'unknown')}")

            # Extract the key part of the selector (the attribute value)
            # [data-test='sidebar-nav-item-nav_item_teststeps'] → sidebar-nav-item-nav_item_teststeps
            import re
            match = re.search(r"'([^']+)'", selector)
            if match:
                selector_value = match.group(1)
            else:
                selector_value = selector

            # Check if selector appears in any visible element
            found = False
            for elem in visible:
                if selector_value in elem or selector in elem:
                    found = True
                    logger.info(f"Selector found in visible elements: {elem}")
                    break

            if found:
                # Selector is visible, accept it with original confidence
                return {
                    'selector': selector,
                    'confidence': min(confidence + 0.15, 1.0),  # Boost confidence if visible
                    'agent': 'L1',
                    'reasoning': 'Selector found in visible elements'
                }
            else:
                # Selector not visible - but could be hidden/offscreen
                # Check if step involves UI elements that might not be in data-* attributes
                ui_element_keywords = ['accordion', 'expand', 'collapse', 'panel', 'dropdown',
                                       'menu', 'modal', 'dialog', 'popup', 'toggle']
                is_ui_element = any(keyword in step_text.lower() for keyword in ui_element_keywords)

                if is_ui_element and confidence > 0.60:
                    # For UI elements with decent confidence, boost to pass threshold
                    logger.info(f"UI element step detected (accordion/dropdown/etc), boosting confidence")
                    return {
                        'selector': selector,
                        'confidence': max(confidence + 0.10, 0.72),  # Boost and ensure >= 0.72
                        'agent': 'L1',
                        'reasoning': 'UI element selector - boosted for accordion/dropdown/panel'
                    }
                else:
                    # For other cases, lower confidence more
                    logger.warning(f"Selector not found in {len(visible)} visible elements")
                    return {
                        'selector': selector,
                        'confidence': max(confidence - 0.10, 0.50),  # Larger penalty
                        'agent': 'L1',
                        'reasoning': 'Selector not in visible list but attempting anyway'
                    }

        except Exception as e:
            logger.warning(f"SelectorAgent_L1: Validation failed, using base confidence: {e}")
            return {
                'selector': selector,
                'confidence': confidence,
                'agent': 'L1',
                'reasoning': 'Validation skipped'
            }


# ============================================================================
# SELECTOR AGENT L2 - DOM Discovery + LLM Analysis
# ============================================================================

class SelectorAgentL2(BaseAgent):
    """Live DOM scraping with LLM analysis"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'selector_agent_l2')
        self.activation_threshold = self.agent_config.get('activation_threshold', 0.70)

    def discover_selector(self, page: Page, step_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover selector from live DOM

        Args:
            page: Playwright page object
            step_text: Step description
            context: Current context

        Returns:
            Discovery result
        """
        logger.info(f"SelectorAgent_L2: Discovering from live DOM for '{step_text}'")

        # Step 1: Scrape DOM elements
        dom_elements = self._scrape_dom_elements(page)

        if not dom_elements:
            logger.warning("SelectorAgent_L2: No DOM elements found")
            return {'selector': None, 'confidence': 0.0, 'agent': 'L2', 'reasoning': 'No DOM elements'}

        # Step 2: LLM analyzes elements and suggests selector
        result = self._analyze_with_llm(dom_elements, step_text, context)

        logger.info(f"SelectorAgent_L2: Result - {result.get('selector')} (conf: {result.get('confidence', 0):.2f})")

        return result

    def _scrape_dom_elements(self, page: Page) -> List[Dict[str, Any]]:
        """Scrape visible interactive elements from page"""
        try:
            elements = page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('button, input, select, a, [role="button"]');
                    return Array.from(elements)
                        .filter(el => el.offsetParent !== null)
                        .slice(0, 100)
                        .map((el, idx) => ({
                            index: idx,
                            tag: el.tagName.toLowerCase(),
                            text: el.textContent.trim().substring(0, 50),
                            attributes: Array.from(el.attributes).reduce((acc, attr) => {
                                acc[attr.name] = attr.value;
                                return acc;
                            }, {}),
                            has_data_attr: Array.from(el.attributes).some(a => a.name.startsWith('data-'))
                        }));
                }
            """)

            logger.info(f"SelectorAgent_L2: Scraped {len(elements)} DOM elements")
            return elements

        except Exception as e:
            logger.error(f"SelectorAgent_L2: DOM scraping failed: {e}")
            return []

    def _analyze_with_llm(self, dom_elements: List[Dict[str, Any]], step_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM analyzes DOM elements and generates selector"""
        try:
            # Format elements for LLM
            elements_str = ""
            for el in dom_elements[:15]:  # Limit to 15
                attrs = el.get('attributes', {})
                data_attrs = {k: v for k, v in attrs.items() if k.startswith('data-')}
                elements_str += f"{el['index']}. <{el['tag']}> text='{el['text']}' data-attrs={data_attrs}\n"

            prompt = f"""
TASK: Identify target element from DOM and generate selector.

STEP: {step_text}

FOUND ELEMENTS ON PAGE:
{elements_str}

PRIORITY: data-* > aria-* > id > class > text-based

Return JSON:
{{
  "selected_element_index": 0-14 or null,
  "selector": "...",
  "confidence": 0.0-1.0,
  "reasoning": "..."
}}
"""

            response = self.call_llm(self.agent_config.get('system_prompt', ''), prompt)

            # Parse JSON
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)

            return {
                'selector': result.get('selector'),
                'confidence': result.get('confidence', 0.0),
                'agent': 'L2',
                'reasoning': result.get('reasoning', '')
            }

        except Exception as e:
            logger.error(f"SelectorAgent_L2: LLM analysis failed: {e}")
            return {'selector': None, 'confidence': 0.0, 'agent': 'L2', 'reasoning': str(e)}


# ============================================================================
# SELECTOR AGENT L3 - Vision Agent (Screenshot + GPT-4 Vision)
# ============================================================================

class SelectorAgentL3(BaseAgent):
    """Vision-based selector discovery using GPT-4 Vision"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'selector_agent_l3')
        self.vision_model = config.get('azure_openai', {}).get('models', {}).get('vision', 'gpt-4o')

    def verify_text(self, page: Page, expected_text: str, step_text: str, screenshot_base64: str = None) -> Dict[str, Any]:
        """
        Verify if expected text exists on page using vision

        Args:
            page: Playwright page object
            expected_text: Text to find
            step_text: Original step description
            screenshot_base64: Optional pre-captured screenshot from previous step

        Returns:
            Verification result with found status
        """
        logger.info(f"SelectorAgent_L3: Verifying text '{expected_text}'")

        try:
            # Use provided screenshot or take new one
            if screenshot_base64:
                logger.info("Using pre-captured screenshot from previous step")
            else:
                import base64
                screenshot_bytes = page.screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                logger.info("Taking new screenshot for verification")

            # Build vision prompt
            prompt = f"""
You are verifying if specific text or information is displayed on this web page screenshot.

EXPECTED TEXT:
"{expected_text}"

TASK:
Determine if this is a SUCCESS MESSAGE verification or a DISPLAY verification:

A) If verifying SUCCESS MESSAGE (contains words like "successfully", "edited", "created", "deleted"):
   1. Look for success messages, notifications, banners, or toasts
   2. Check if the message SEMANTICALLY MATCHES the expected text
   3. Focus on key information: action (edited/created/deleted), entity type, and name

   EXAMPLES:
   - Expected: "Successfully edited: 'TestObject' default_testobject_01"
   - Valid: "Successfully edited TestObject: default_testobject_01" ✓

B) If verifying DISPLAY (contains words like "displayed", "is visible", "appears", "verify name"):
   1. Look ANYWHERE on the page (headers, titles, labels, tables, forms, detail panels)
   2. Check if the specified text/name/attribute is VISIBLE on the page
   3. Match based on MEANING and KEY IDENTIFIERS

   EXAMPLES:
   - Expected: "teststep name default_Measurement01 is displayed in details header"
   - Valid: Found "default_Measurement01" in page title or header ✓
   - Expected: "Verify Name attribute is displayed"
   - Valid: Found "Name" field or label in details section ✓

IMPORTANT: Match based on MEANING, not exact wording. Variations in quotes, punctuation, or word order are acceptable.

Return JSON only:
{{
  "found": true/false,
  "actual_text": "exact text you found" or null,
  "location": "where you found it (e.g., 'header', 'details panel', 'green banner')" or null,
  "confidence": 0.0-1.0,
  "reasoning": "explanation of match or why not found"
}}
"""

            # Call GPT-4 Vision
            response = self.azure_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON
            if '```json' in result_text:
                json_str = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                json_str = result_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = result_text

            result = json.loads(json_str)

            logger.info(f"SelectorAgent_L3: Text found={result.get('found')}, confidence={result.get('confidence', 0):.2f}")

            return {
                'found': result.get('found', False),
                'actual_text': result.get('actual_text'),
                'location': result.get('location'),
                'confidence': result.get('confidence', 0.0),
                'agent': 'L3',
                'reasoning': result.get('reasoning', '')
            }

        except Exception as e:
            logger.error(f"SelectorAgent_L3: Vision verification failed: {e}")
            # Fallback: Simple text search in page body
            try:
                body_text = page.text_content('body')
                found = expected_text.lower() in body_text.lower()
                logger.info(f"SelectorAgent_L3: Fallback text search found={found}")
                return {
                    'found': found,
                    'actual_text': expected_text if found else None,
                    'location': 'page body' if found else None,
                    'confidence': 0.8 if found else 0.0,
                    'agent': 'L3_fallback',
                    'reasoning': f'Fallback text search: {e}'
                }
            except:
                return {
                    'found': False,
                    'actual_text': None,
                    'location': None,
                    'confidence': 0.0,
                    'agent': 'L3_fallback',
                    'reasoning': f'Vision and fallback failed: {e}'
                }


# ============================================================================
# ORCHESTRATOR AGENT - Decision Maker & Router
# ============================================================================

class OrchestratorAgent(BaseAgent):
    """Routes to appropriate agents and logs decisions"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'orchestrator_agent')

    def decide_next_agent(self, step_text: str, context: Dict[str, Any],
                          state: TestExecutionState, learned_result: Optional[Dict] = None,
                          l1_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Decide which agent to use next

        Args:
            step_text: Step description
            context: Current context
            state: Execution state
            learned_result: Result from LearningAgent
            l1_result: Result from L1

        Returns:
            Decision with next_agent and reasoning
        """
        # Decision logic
        if learned_result and learned_result.get('confidence', 0) > 0.90:
            decision = {
                'next_agent': 'execute',
                'use_selector': learned_result['selector'],
                'confidence': learned_result['confidence'],
                'reasoning': f"High-confidence learned selector found ({learned_result['confidence']:.2f})"
            }
        elif l1_result and l1_result.get('confidence', 0) >= 0.70:
            decision = {
                'next_agent': 'execute',
                'use_selector': l1_result['selector'],
                'confidence': l1_result['confidence'],
                'reasoning': f"L1 RAG selector validated by LLM ({l1_result['confidence']:.2f})"
            }
        elif l1_result:
            decision = {
                'next_agent': 'l2',
                'reasoning': f"L1 confidence too low ({l1_result.get('confidence', 0):.2f}), trying L2 DOM discovery"
            }
        else:
            decision = {
                'next_agent': 'l1',
                'reasoning': "Starting with L1 RAG search"
            }

        # Log decision
        logger.info(f"OrchestratorAgent: {decision['reasoning']}")
        state['orchestrator_reasoning'].append(f"Step {state['current_step']}: {decision['reasoning']}")

        return decision


# ============================================================================
# PLCD TESTING ASSISTANT - SEQUENTIAL VERSION
# ============================================================================

class PLCDTestingAssistantSeq:
    """
    Sequential context tracking version with LangGraph agents
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize testing assistant

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize agents
        self.jira_agent = JiraAgent(config)
        self.context_agent = ContextAgent(config)
        self.learning_agent = LearningAgent(config)
        self.selector_agent_l1 = SelectorAgentL1(config)
        self.selector_agent_l2 = SelectorAgentL2(config)
        self.selector_agent_l3 = SelectorAgentL3(config)
        self.orchestrator_agent = OrchestratorAgent(config)

        logger.info("PLCD Testing Assistant (Sequential) initialized")

    def execute_test(self, ticket_id: str) -> Dict[str, Any]:
        """
        Execute test for a Jira ticket

        Args:
            ticket_id: Jira ticket ID

        Returns:
            Execution results
        """
        print("\n" + "=" * 80)
        print(f"PLCD Testing Assistant (Sequential) - Executing: {ticket_id}")
        print("=" * 80)

        # Initialize state
        state = self._initialize_state(ticket_id)

        try:
            # Step 1: Parse Jira ticket with LLM
            ticket_data = self._parse_ticket_llm(state)

            # Step 2: Initialize browser
            with sync_playwright() as playwright:
                browser = self._init_browser(playwright)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1200},
                    record_video_dir="Videos/" if self.config.get('artifacts', {}).get('video_recording', {}).get('enabled', True) else None
                )
                page = context.new_page()

                # Step 3: Capture initial context
                self._capture_initial_context(state, page)

                # Step 4: Login
                self._login(state, page)

                # Step 5: Execute test steps with agents
                self._execute_steps_with_agents(state, page, ticket_data)

                # Get video path
                if self.config.get('artifacts', {}).get('video_recording', {}).get('enabled', True):
                    try:
                        state['video_path'] = str(page.video.path()) if page.video else None
                    except:
                        state['video_path'] = None

                # Cleanup
                page.close()
                browser.close()

            # Step 6: Generate artifacts
            results = self._generate_artifacts(state)

            print("\n" + "=" * 80)
            print("Test Execution Complete!")
            print("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            state['overall_status'] = "FAILED"
            print(f"\n[ERROR] Test execution failed: {e}")
            return self._generate_artifacts(state)

    def _initialize_state(self, ticket_id: str) -> TestExecutionState:
        """Initialize execution state"""
        state: TestExecutionState = {
            'ticket_id': ticket_id,
            'ticket_title': '',
            'module': '',
            'current_step': 0,
            'total_steps': 0,
            'context_history': [],
            'current_context': {},
            'selector_attempts': [],
            'successful_selectors': {},
            'failure_patterns': [],
            'corrections_used': [],
            'agent_chain': [],
            'next_agent': 'jira',
            'orchestrator_reasoning': [],
            'step_results': [],
            'overall_status': 'PASSED',
            'video_path': None,
            'script_path': None,
            'report_path': None,
            'execution_start_time': time.time(),
            'config': self.config
        }
        return state

    def _parse_ticket_llm(self, state: TestExecutionState) -> Dict[str, Any]:
        """Parse ticket using JiraAgent"""
        print("\n[1/6] Parsing Jira ticket with LLM...")

        ticket_data = self.jira_agent.parse_ticket(state['ticket_id'])

        state['ticket_title'] = ticket_data.get('title', '')
        state['module'] = ticket_data.get('module', '')
        state['total_steps'] = len(ticket_data.get('steps', []))
        state['agent_chain'].append('JiraAgent')

        print(f"[OK] Ticket: {ticket_data.get('title', 'N/A')}")
        print(f"[OK] Module: {state['module']}")
        print(f"[OK] Steps: {state['total_steps']}")

        return ticket_data

    def _init_browser(self, playwright) -> Browser:
        """Initialize browser"""
        print("\n[2/6] Initializing browser...")

        browser_type = self.config['browser']
        headless = self.config['execution']['headless']

        if browser_type == 'edge':
            browser = playwright.chromium.launch(
                headless=headless,
                channel='msedge',
                args=['--start-maximized', '--window-size=1920,1200']
            )
        else:
            browser = playwright.chromium.launch(
                headless=headless,
                args=['--start-maximized', '--window-size=1920,1200']
            )

        print(f"[OK] Browser: {browser_type}")
        return browser

    def _capture_initial_context(self, state: TestExecutionState, page: Page):
        """Capture initial context"""
        print("\n[3/6] Capturing initial context...")

        page.goto(self.config['web_url'])
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(2000)

        # Capture context
        context = self.context_agent.capture_context(page)
        self.context_agent.update_history(state, context)
        state['agent_chain'].append('ContextAgent')

        print(f"[OK] Initial context captured")

    def _login(self, state: TestExecutionState, page: Page):
        """Login to application"""
        print("\n[4/6] Logging in...")

        username = self.config['login']['username']
        password = self.config['login']['password']

        try:
            page.fill('input[type="text"]', username)
            page.fill('input[type="password"]', password)

            # TODO: Use selector agent for login button
            # For now, use simple selector
            page.locator('button').first.click()

            page.wait_for_timeout(self.config['wait_times']['after_login'])

            # Capture post-login context
            context = self.context_agent.capture_context(page)
            self.context_agent.update_history(state, context)

            # Set logged_in flag
            state['logged_in'] = True

            print(f"[OK] Logged in as: {username}")

        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise

    def _execute_steps_with_agents(self, state: TestExecutionState, page: Page, ticket_data: Dict[str, Any]):
        """Execute test steps using Learning + Selector agents"""
        print("\n[5/6] Executing test steps with agents...")
        print("-" * 80)

        steps = ticket_data.get('steps', [])

        for step_data in steps:
            step_number = step_data.get('number', 0)
            step_text = step_data.get('text', '')
            state['current_step'] = step_number

            print(f"\nStep {step_number}/{len(steps)}: {step_text}")

            # Skip "Login" step if already logged in
            if step_text.lower().strip() in ['login', 'log in', 'sign in', 'signin']:
                logger.info(f"Detected login step. logged_in flag: {state.get('logged_in', False)}")
                if state.get('logged_in', False):
                    print(f"  [SKIPPED] Already logged in")

                    # Ensure we're on dashboard after skipping login
                    current_url = page.url
                    if '/login' in current_url and '/dashboard' not in current_url:
                        logger.info("Still on login page, navigating to dashboard")
                        dashboard_url = current_url.replace('/login', '/dashboard')
                        page.goto(dashboard_url)
                        page.wait_for_load_state('networkidle')

                        # Recapture context after navigation
                        context = self.context_agent.capture_context(page)
                        self.context_agent.update_history(state, context)
                        logger.info("Context updated after dashboard navigation")

                    state['step_results'].append({
                        'step': step_number,
                        'text': step_text,
                        'status': 'skipped',
                        'selector': None,
                        'confidence': 1.0,
                        'agent': 'N/A',
                        'reason': 'Already logged in'
                    })
                    continue

            try:
                # Capture context before step
                context = self.context_agent.capture_context(page)
                self.context_agent.update_history(state, context)

                # Step 0: Check if this is a text verification step (use L3 directly)
                if self._is_text_verification_step(step_text):
                    expected_text = self._extract_expected_text(step_text)
                    if expected_text:
                        logger.info(f"Text verification step detected, using L3 Vision Agent")
                        print(f"  [L3 Vision] Verifying text: '{expected_text}'")

                        # Use screenshot from previous step if available
                        prev_screenshot = state.get('last_screenshot')
                        l3_result = self.selector_agent_l3.verify_text(page, expected_text, step_text, prev_screenshot)
                        state['agent_chain'].append('SelectorAgent_L3')

                        success = l3_result['found']
                        selector = f"Text verification: '{expected_text}'"
                        confidence = l3_result['confidence']
                        agent_used = 'L3'

                        if success:
                            print(f"  [L3] Text found: {l3_result.get('actual_text', expected_text)}")
                            print(f"       Location: {l3_result.get('location', 'unknown')}")
                            print(f"       Confidence: {confidence:.2f}")
                        else:
                            print(f"  [L3] Text NOT found (conf: {confidence:.2f})")
                            print(f"       Reason: {l3_result.get('reasoning', 'unknown')}")

                        # Record result
                        step_result = {
                            'step_number': step_number,
                            'step_text': step_text,
                            'selector': selector,
                            'confidence': confidence,
                            'agent_used': agent_used,
                            'action_type': 'verify_text',
                            'status': 'PASSED' if success else 'FAILED',
                            'context': context,
                            'l3_result': l3_result
                        }

                        state['step_results'].append(step_result)

                        if success:
                            print(f"[OK] Text verified (agent: L3)")
                        else:
                            print(f"[FAILED] Text verification failed")
                            state['overall_status'] = 'FAILED'

                            if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                                print(f"\n[FAIL-FAST] Stopping execution")
                                break

                        continue  # Skip to next step

                # Regular selector-based steps
                # Step 1: Check LearningAgent for known selector
                # TEMPORARILY DISABLED - Learning Agent confuses edit/delete buttons due to visible elements in query
                learned = None  # self.learning_agent.query_learned_selector(step_text, context)

                if learned and learned['confidence'] > 0.90:
                    # Use learned selector
                    selector = learned['selector']
                    confidence = learned['confidence']
                    agent_used = 'Learning'
                    print(f"  [Learning] Using learned selector: {selector} (conf: {confidence:.2f})")
                    state['agent_chain'].append('LearningAgent')

                else:
                    # Step 2: Try SelectorAgent_L1 (RAG + LLM)
                    l1_result = self.selector_agent_l1.discover_selector(step_text, context, state)
                    state['agent_chain'].append('SelectorAgent_L1')

                    if l1_result['selector'] and l1_result['confidence'] >= 0.70:
                        # Use L1 result
                        selector = l1_result['selector']
                        confidence = l1_result['confidence']
                        agent_used = 'L1'
                        print(f"  [L1] {selector} (conf: {confidence:.2f})")

                    else:
                        # Step 3: Try SelectorAgent_L2 (DOM + LLM)
                        print(f"  [L1 Low Confidence: {l1_result['confidence']:.2f}] Trying L2...")
                        l2_result = self.selector_agent_l2.discover_selector(page, step_text, context)
                        state['agent_chain'].append('SelectorAgent_L2')

                        if l2_result['selector'] and l2_result['confidence'] >= 0.70:
                            selector = l2_result['selector']
                            confidence = l2_result['confidence']
                            agent_used = 'L2'
                            print(f"  [L2] {selector} (conf: {confidence:.2f})")
                        else:
                            # No selector found
                            raise Exception(f"No selector found (L1: {l1_result['confidence']:.2f}, L2: {l2_result['confidence']:.2f})")

                # Execute action
                action_type = self._detect_action_type(step_text)
                success = self._execute_action(page, selector, action_type, step_text, state)

                # Record result
                step_result = {
                    'step_number': step_number,
                    'step_text': step_text,
                    'selector': selector,
                    'confidence': confidence,
                    'agent_used': agent_used,
                    'action_type': action_type,
                    'status': 'PASSED' if success else 'FAILED',
                    'context': context
                }

                state['step_results'].append(step_result)

                if success:
                    print(f"[OK] {selector} (agent: {agent_used})")

                    # Store to learning for future use
                    self.learning_agent.store_learned_selector(
                        step_text, selector, context, confidence, source='runtime', verified=True
                    )
                else:
                    print(f"[FAILED] Execution failed")
                    state['overall_status'] = 'FAILED'

                    if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                        print(f"\n[FAIL-FAST] Stopping execution")
                        break

            except Exception as e:
                logger.error(f"Step {step_number} failed: {e}")
                state['step_results'].append({
                    'step_number': step_number,
                    'step_text': step_text,
                    'status': 'FAILED',
                    'error': str(e),
                    'context': context
                })
                state['overall_status'] = 'FAILED'
                print(f"[FAILED] Error: {e}")

                if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                    print(f"\n[FAIL-FAST] Stopping execution")
                    break

        print("-" * 80)

    def _is_text_verification_step(self, step_text: str) -> bool:
        """Check if step requires text verification using L3 Vision"""
        step_lower = step_text.lower()

        verify_keywords = ['verify', 'check', 'validate', 'confirm']
        text_keywords = ['message', 'text', 'displayed', 'shown', 'appears', 'contains']

        has_verify = any(kw in step_lower for kw in verify_keywords)
        has_text = any(kw in step_lower for kw in text_keywords)

        return has_verify and has_text

    def _extract_expected_text(self, step_text: str) -> Optional[str]:
        """Extract expected text from verification step"""
        import re

        # Pattern 1: "Verify success message <text> is displayed"
        # Extract everything between "message" and "is displayed/shown"
        message_pattern = r'(?:verify|check).*?message\s+(.+?)\s+(?:is|are)\s+(?:displayed|shown)'
        match = re.search(message_pattern, step_text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            # Remove surrounding quotes if present
            text = text.strip('"').strip("'")
            return text

        # Pattern 2: Try to find text in quotes (single or double)
        patterns = [
            r'"([^"]+)"',  # Double quotes
            r"'([^']+)'",  # Single quotes
        ]

        for pattern in patterns:
            match = re.search(pattern, step_text)
            if match:
                return match.group(1).strip()

        # Pattern 3: After "message:" or "text:"
        colon_pattern = r'(?:message|text)[:\s]+([^\.]+)'
        match = re.search(colon_pattern, step_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: extract text after "verify" or "check"
        verify_match = re.search(r'(?:verify|check)\s+(.+?)(?:is|are|displayed|shown)', step_text, re.IGNORECASE)
        if verify_match:
            return verify_match.group(1).strip()

        return None

    def _detect_action_type(self, step_text: str) -> str:
        """Detect action type from step text"""
        step_lower = step_text.lower()

        if any(word in step_lower for word in ['click', 'press', 'select', 'choose']):
            return 'click'
        elif any(word in step_lower for word in ['enter', 'type', 'input', 'fill']):
            return 'type'
        elif 'navigate' in step_lower or 'go to' in step_lower:
            return 'navigate'
        elif 'verify' in step_lower or 'check' in step_lower or 'wait' in step_lower:
            return 'verify'
        elif 'clear' in step_lower:
            return 'clear'
        else:
            return 'click'  # Default

    def _execute_action(self, page: Page, selector: str, action_type: str, step_text: str, state: TestExecutionState = None) -> bool:
        """Execute Playwright action and capture screenshot after"""
        try:
            page.wait_for_selector(selector, timeout=self.config['performance']['element_wait_timeout'])

            # Check if this is a save/submit action
            step_lower = step_text.lower()
            is_save_action = any(word in step_lower for word in ['save', 'submit', 'create', 'update', 'delete'])

            if action_type == 'click':
                page.click(selector)
                # Use longer wait for save/submit actions to allow success messages to appear
                if is_save_action and 'after_save' in self.config['wait_times']:
                    page.wait_for_timeout(self.config['wait_times']['after_save'])
                else:
                    page.wait_for_timeout(self.config['wait_times']['after_click'])

            elif action_type == 'type':
                text_to_type = self._extract_text_to_type(step_text)
                page.fill(selector, text_to_type)
                page.wait_for_timeout(self.config['wait_times']['after_type'])

            elif action_type == 'clear':
                page.fill(selector, '')
                page.wait_for_timeout(self.config['wait_times']['after_type'])

            elif action_type == 'verify':
                page.wait_for_timeout(self.config['wait_times']['after_click'])

            elif action_type == 'navigate':
                page.click(selector)
                page.wait_for_timeout(self.config['wait_times']['after_navigation'])

            # Capture screenshot after action for next step's verification
            if state is not None:
                import base64
                screenshot_bytes = page.screenshot()
                state['last_screenshot'] = base64.b64encode(screenshot_bytes).decode('utf-8')
                logger.info(f"Captured post-action screenshot ({len(screenshot_bytes)} bytes)")

            return True

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    def _extract_text_to_type(self, step_text: str) -> str:
        """Extract text to type from step description"""
        import re
        match = re.search(r'"([^"]+)"', step_text)
        if match:
            return match.group(1)
        match = re.search(r"'([^']+)'", step_text)
        if match:
            return match.group(1)
        return "Test Value"

    def _generate_artifacts(self, state: TestExecutionState) -> Dict[str, Any]:
        """Generate execution artifacts"""
        print("\n[6/6] Generating artifacts...")

        execution_time = time.time() - state['execution_start_time']

        passed_steps = sum(1 for r in state['step_results'] if r.get('status') == 'PASSED')
        failed_steps = sum(1 for r in state['step_results'] if r.get('status') == 'FAILED')

        summary = {
            "ticket_id": state['ticket_id'],
            "overall_status": state['overall_status'],
            "total_steps": len(state['step_results']),
            "passed_steps": passed_steps,
            "failed_steps": failed_steps,
            "execution_time": f"{execution_time:.1f}s",
            "step_results": state['step_results'],
            "agent_chain": state['agent_chain'],
            "video_path": state.get('video_path'),
            "context_history": state['context_history']
        }

        # Generate HTML report (TODO: enhance with context trace)
        try:
            ticket_data = {
                'title': state['ticket_title'],
                'module': state['module'],
                'steps': state['step_results']
            }

            report_path = generate_html_report(
                ticket_id=state['ticket_id'],
                ticket_data=ticket_data,
                step_results=state['step_results'],
                overall_status=state['overall_status'],
                execution_time=execution_time,
                config=self.config
            )

            summary['report_path'] = report_path
            print(f"[OK] HTML Report: {report_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            print(f"[WARNING] Report generation failed: {e}")

        # Generate Playwright script (only for successful tests)
        if state['overall_status'] == 'PASSED':
            try:
                script_path = generate_playwright_script(
                    ticket_id=state['ticket_id'],
                    ticket_data=ticket_data,
                    step_results=state['step_results'],
                    config=self.config
                )
                summary['script_path'] = script_path
                print(f"[OK] Playwright Script: {script_path}")
            except Exception as e:
                logger.error(f"Script generation failed: {e}")
                print(f"[WARNING] Script generation failed: {e}")

        # Save context trace (debug)
        try:
            context_trace_path = Path("Logs") / f"context_trace_{state['ticket_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(context_trace_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'ticket_id': state['ticket_id'],
                    'context_history': state['context_history'],
                    'agent_chain': state['agent_chain']
                }, f, indent=2)
            print(f"[OK] Context Trace: {context_trace_path}")
        except Exception as e:
            logger.error(f"Context trace save failed: {e}")

        print(f"\n[OK] Summary:")
        print(f"     Status: {summary['overall_status']}")
        print(f"     Total Steps: {summary['total_steps']}")
        print(f"     Passed: {summary['passed_steps']}")
        print(f"     Failed: {summary['failed_steps']}")
        print(f"     Execution Time: {summary['execution_time']}")
        print(f"     Agent Chain: {' -> '.join(state['agent_chain'])}")

        return summary


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python plcd_taseq.py <TICKET_ID>")
        print("Example: python plcd_taseq.py RBPLCD-8835")
        sys.exit(1)

    ticket_id = sys.argv[1]

    try:
        # Load configuration
        config = load_config()

        # Initialize assistant
        assistant = PLCDTestingAssistantSeq(config)

        # Execute test
        results = assistant.execute_test(ticket_id)

        # Exit with appropriate code
        sys.exit(0 if results['overall_status'] == 'PASSED' else 1)

    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
