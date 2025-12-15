"""
PLCD Testing Assistant - Sequential Context Tracking Version
Uses LangGraph multi-agent architecture for intelligent test execution
"""

import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime

from playwright.sync_api import sync_playwright, Page, Browser

from config_loader import load_config, get_azure_client
from report_generator import generate_html_report
from script_generator import generate_playwright_script, generate_pytest_config, generate_readme

# Setup logging with environment variable control
import os

# LOG_LEVEL: DEBUG (verbose), INFO (clean), WARNING (minimal)
LOG_LEVEL = os.environ.get('PLCD_LOG_LEVEL', 'INFO').upper()

# File handler - always DEBUG (full logs saved)
file_handler = logging.FileHandler('Logs/plcd_taseq.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler - respects LOG_LEVEL (clean for testers)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console_handler.setFormatter(logging.Formatter('%(message)s'))  # Cleaner console format

logging.basicConfig(
    level=logging.DEBUG,  # Root level DEBUG so file gets everything
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding for unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # If reconfigure fails, continue with ASCII-safe output

# Suppress verbose HTTP request logging from Azure OpenAI client
# This prevents base64 screenshot data from appearing in console during L3 Vision API calls
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Suppress ChromaDB verbose warnings and info messages in console
# These still get logged to file, but don't clutter terminal output
logging.getLogger("chromadb").setLevel(logging.ERROR)  # Only show errors
logging.getLogger("chromadb.db.impl.sqlite").setLevel(logging.ERROR)


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

        logger.debug(f"{agent_name} initialized (model: {self.model})")

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
        logger.debug(f"JiraAgent: Parsing ticket {ticket_id}")

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

            logger.debug(f"JiraAgent: Parsed {len(parsed_data.get('steps', []))} steps")
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

            logger.debug(f"ContextAgent: Captured context from {context['pathname']}")
            logger.debug(f"ContextAgent: Found {len(context['visible_elements'])} visible data-* elements")

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

        logger.debug(f"LearningAgent: Collection '{collection_name}' initialized")

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

            logger.debug(f"LearningAgent: Stored selector for '{step_text}'")

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

        # Initialize ChromaDB client for vector search
        import chromadb
        from chromadb.config import Settings
        from pathlib import Path
        chroma_path = Path(config['vector_database']['persist_directory'])
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Store config for vector search functions
        self.config = config

        self.confidence_threshold = self.agent_config.get('confidence_threshold', 0.75)
        self.retry_threshold = self.agent_config.get('retry_threshold', 0.70)

    def discover_selector(self, step_text: str, context: Dict[str, Any], state: TestExecutionState) -> Dict[str, Any]:
        """
        Discover selector using RAG + LLM validation
        NEW: Checks pending insights FIRST before L1/L2/L3

        Args:
            step_text: Step description
            context: Current page context
            state: Execution state

        Returns:
            Discovery result with selector and confidence
        """
        logger.debug(f"SelectorAgent_L1: Discovering selector for '{step_text}'")

        # Step 0: Check pending insights FIRST (fast JSON lookup)
        logger.debug(f"Checking pending insights for: '{step_text}'")
        pending_result = self._check_pending_insights(step_text, state)
        if pending_result:
            logger.info(f"✓ Found pending insight - using learned selector")
            return pending_result
        else:
            logger.debug(f"No pending insight found, proceeding to L1")

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
            logger.debug(f"Navigation query detected, skipping page_context filter")
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

        # Step 2.4: Dynamic selector substitution (SCALABLE - works for all values!)
        # Check if matched selector is dynamic and substitute values
        selector_result = rag_result.get('selector_result', {})
        metadata = selector_result.get('metadata', {})

        if metadata.get('isDynamic', False):
            # This is a dynamic selector - need to substitute template with actual value
            template_pattern = metadata.get('value', '')  # e.g., "{{item}}", "{{name}}", "Type 3"
            current_selector = selector_result.get('selector', '')

            # Extract dynamic value from current step text
            query_value = self._extract_dynamic_value(step_text)  # e.g., "Type 4"

            if query_value and template_pattern:
                # Check if template pattern exists in selector
                if template_pattern in current_selector:
                    # Template-based substitution: {{item}} → Type 4
                    new_selector = current_selector.replace(template_pattern, query_value)

                    logger.info(f"🔄 Dynamic template substitution:")
                    logger.info(f"   Template: '{template_pattern}' → Value: '{query_value}'")
                    logger.info(f"   Old selector: {current_selector}")
                    logger.info(f"   New selector: {new_selector}")
                elif query_value != template_pattern:
                    # Value-based substitution: Type 3 → Type 4 (legacy support)
                    new_selector = current_selector.replace(template_pattern, query_value)

                    logger.info(f"🔄 Dynamic value substitution:")
                    logger.info(f"   Stored: '{template_pattern}' → Query: '{query_value}'")
                    logger.info(f"   Old selector: {current_selector}")
                    logger.info(f"   New selector: {new_selector}")
                else:
                    # Values are identical, no substitution needed
                    new_selector = current_selector
                    logger.debug(f"Dynamic selector values match, no substitution needed")

                # Update the selector in result
                selector_result['selector'] = new_selector
                rag_result['selector_result']['selector'] = new_selector

                # Update all candidates with same substitution
                if rag_result.get('candidates'):
                    for candidate in rag_result['candidates']:
                        if candidate.get('metadata', {}).get('isDynamic'):
                            cand_template = candidate.get('metadata', {}).get('value', '')
                            cand_selector = candidate.get('selector', '')
                            if cand_template and cand_template in cand_selector:
                                candidate['selector'] = cand_selector.replace(cand_template, query_value)

        # Step 2.5: Re-rank candidates if step contains specific names
        # Generic: Extract names from step text and boost selectors containing those names
        import re
        # Find words that look like identifiers with underscores or mixed alphanumeric
        # Pattern: word_word, word_word_01, default_Measurement01, test_object, etc.
        name_patterns = re.findall(r'\b[a-zA-Z]+[_][a-zA-Z0-9_]+\b', step_text)

        if name_patterns and rag_result.get('candidates'):
            logger.debug(f"Detected name patterns in step: {name_patterns}")
            # Check all candidates for name matches
            for candidate in rag_result['candidates']:
                selector = candidate.get('selector', '')
                # Check if selector contains any of the detected names
                for name in name_patterns:
                    if name.lower() in selector.lower():
                        logger.debug(f"Name match found: '{name}' in {selector}, boosting confidence")
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
                logger.debug(f"Detected '{action}' action in step, applying keyword matching")
                for candidate in rag_result['candidates']:
                    selector_lower = candidate.get('selector', '').lower()
                    # Boost matching keywords
                    for boost_word in keywords['boost']:
                        if boost_word in selector_lower:
                            logger.debug(f"Action match: '{boost_word}' in {candidate['selector']}, boosting")
                            candidate['confidence'] = min(candidate['confidence'] + 0.15, 1.0)
                            break
                    # Penalize conflicting keywords
                    for penalize_word in keywords['penalize']:
                        if penalize_word in selector_lower:
                            logger.debug(f"Action conflict: '{penalize_word}' in {candidate['selector']}, penalizing")
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

        logger.debug(f"SelectorAgent_L1: Result - {validated_result['selector']} (conf: {validated_result['confidence']:.2f})")

        return validated_result

    def _check_pending_insights(self, step_text: str, state: TestExecutionState) -> Optional[Dict[str, Any]]:
        """
        Check learned insights using vector search (embedded + pending).

        NEW: Searches in priority order:
        1. Learned insights collection (ChromaDB - fast, embedded)
        2. Pending insights folder (JSON files - slower, LLM comparison)

        Args:
            step_text: Step description
            state: Execution state

        Returns:
            Selector result if found, None otherwise
        """
        try:
            current_module = state.get('module', '')

            # Priority 1: Search learned insights (embedded in ChromaDB)
            logger.debug(f"Searching learned insights for: '{step_text}'")
            learned_matches = search_learned_insights(
                self.chroma_client,
                self.azure_client,
                self.config,
                step_text,
                current_module
            )

            if learned_matches:
                best_match = learned_matches[0]  # Already sorted by confidence
                logger.info(f"✓ Learned Insight Found (Embedded):")
                logger.info(f"  Selector: {best_match['selector']}")
                logger.info(f"  Confidence: {best_match['confidence']:.2f} (similarity: {best_match['similarity']:.2f})")
                logger.info(f"  Source: {best_match['source']}, Category: {best_match['category']}")

                return {
                    'selector': best_match['selector'],
                    'confidence': best_match['confidence'],
                    'agent': 'Learned Insight (Embedded)',
                    'reasoning': f"Found via semantic search (similarity: {best_match['similarity']:.2f})"
                }

            # Priority 2: Search pending insights (Embedding-based - fast and precise)
            logger.debug(f"No embedded insights found, searching pending insights with embeddings...")
            pending_matches = search_pending_insights_embeddings(
                self.azure_client,
                self.config,
                step_text,
                current_module
            )

            if pending_matches:
                best_match = pending_matches[0]  # Already sorted by confidence
                logger.info(f"✓ Pending Insight Found (Embeddings):")
                logger.info(f"  Selector: {best_match['selector']}")
                logger.info(f"  Confidence: {best_match['confidence']:.2f} (similarity: {best_match['similarity']:.2f})")
                logger.info(f"  File: {best_match.get('file', 'N/A')}")

                # Apply dynamic selector substitution if needed
                selector = best_match['selector']
                metadata = best_match.get('metadata', {})

                if metadata.get('isDynamic', False):
                    # This is a dynamic selector - substitute template with actual value
                    template_value = metadata.get('value', '')  # e.g., "Type 3"

                    # Extract dynamic value from current step text
                    query_value = self._extract_dynamic_value(step_text)

                    if query_value and template_value:
                        # Value-based substitution: Type 3 → Type 1
                        if query_value != template_value and template_value in selector:
                            new_selector = selector.replace(template_value, query_value)

                            logger.info(f"🔄 Dynamic value substitution:")
                            logger.info(f"   Stored: '{template_value}' → Query: '{query_value}'")
                            logger.info(f"   Old selector: {selector}")
                            logger.info(f"   New selector: {new_selector}")

                            selector = new_selector
                        else:
                            logger.debug(f"Dynamic selector values match or template not in selector, no substitution needed")

                return {
                    'selector': selector,
                    'confidence': best_match['confidence'],
                    'agent': 'Pending Insight (Embeddings)',
                    'reasoning': f"Found via embedding similarity: {best_match['similarity']:.2f}"
                }

            logger.debug("No learned or pending insights found")
            return None

        except Exception as e:
            logger.error(f"Error checking learned/pending insights: {e}", exc_info=True)
            return None

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

            logger.debug(f"SelectorAgent_L1: Enhanced query: '{enhanced_query}'")
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
            logger.debug(f"Validating selector {selector} against {len(visible)} visible elements on page {context.get('url', 'unknown')}")

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
                    logger.debug(f"Selector found in visible elements: {elem}")
                    break

            if found:
                # Selector is visible, accept it with original confidence
                return {
                    'selector': selector,
                    'confidence': min(confidence + 0.15, 1.0),  # Boost confidence if visible
                    'agent': 'L1',
                    'reasoning': 'Selector found in visible elements',
                    'selector_validated': True  # Selector IS on the page
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
                        'reasoning': 'UI element selector - boosted for accordion/dropdown/panel',
                        'selector_validated': True  # Trust UI element selectors
                    }
                else:
                    # For other cases, lower confidence more
                    logger.debug(f"Selector not found in {len(visible)} visible elements")
                    return {
                        'selector': selector,
                        'confidence': max(confidence - 0.10, 0.50),  # Larger penalty
                        'agent': 'L1',
                        'reasoning': 'Selector not in visible list but attempting anyway',
                        'selector_validated': False  # Selector NOT on the page - should try L2!
                    }

        except Exception as e:
            logger.warning(f"SelectorAgent_L1: Validation failed, using base confidence: {e}")
            return {
                'selector': selector,
                'confidence': confidence,
                'agent': 'L1',
                'reasoning': 'Validation skipped',
                'selector_validated': False  # Validation failed - try L2
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
        logger.debug(f"SelectorAgent_L2: Discovering from live DOM for '{step_text}'")

        # Step 1: Scrape DOM elements
        dom_elements = self._scrape_dom_elements(page)

        if not dom_elements:
            logger.warning("SelectorAgent_L2: No DOM elements found")
            return {'selector': None, 'confidence': 0.0, 'agent': 'L2', 'reasoning': 'No DOM elements'}

        # Step 2: LLM analyzes elements and suggests selector
        result = self._analyze_with_llm(dom_elements, step_text, context)

        logger.debug(f"SelectorAgent_L2: Result - {result.get('selector')} (conf: {result.get('confidence', 0):.2f})")

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

            logger.debug(f"SelectorAgent_L2: Scraped {len(elements)} DOM elements")
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

        logger.debug("PLCD Testing Assistant (Sequential) initialized")

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
                    viewport=None,  # Use full window size (no fixed viewport)
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

                # Rename video file to match report/script naming convention
                if state.get('video_path'):
                    try:
                        import shutil
                        old_video_path = Path(state['video_path'])
                        if old_video_path.exists():
                            # Generate new filename with same pattern as report/script
                            video_folder = Path(self.config['folders']['videos'])
                            video_folder.mkdir(parents=True, exist_ok=True)
                            new_video_filename = f"{state['ticket_id']}_{state['timestamp']}.webm"
                            new_video_path = video_folder / new_video_filename

                            # Move/rename video file
                            shutil.move(str(old_video_path), str(new_video_path))
                            state['video_path'] = str(new_video_path)
                            logger.debug(f"Video renamed: {new_video_filename}")
                    except Exception as e:
                        logger.warning(f"Video rename failed: {e}")

            # Step 6: Generate artifacts
            results = self._generate_artifacts(state)

            # IMPORTANT: Cleanup ChromaDB connections
            self._cleanup_chroma_connections()

            print("\n" + "=" * 80)
            print("Test Execution Complete!")
            print("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            state['overall_status'] = "FAILED"
            print(f"\n[ERROR] Test execution failed: {e}")
            # Cleanup even on error
            self._cleanup_chroma_connections()
            return self._generate_artifacts(state)

    def _cleanup_chroma_connections(self):
        """
        Cleanup ChromaDB connections to release file locks.
        CRITICAL: This prevents database corruption and file locking issues.
        """
        try:
            # Close ChromaDB client connections in agents
            if hasattr(self, 'learning_agent') and hasattr(self.learning_agent, 'chroma_client'):
                # ChromaDB doesn't have explicit close(), but clearing references helps GC
                self.learning_agent.chroma_client = None
                self.learning_agent.learning_collection = None
                logger.debug("LearningAgent ChromaDB connection cleared")

            if hasattr(self, 'selector_agent_l1') and hasattr(self.selector_agent_l1, 'chroma_client'):
                self.selector_agent_l1.chroma_client = None
                logger.debug("SelectorAgent_L1 ChromaDB connection cleared")

            # Force garbage collection to release file handles
            import gc
            gc.collect()
            logger.debug("ChromaDB cleanup completed")

        except Exception as e:
            logger.warning(f"Error during ChromaDB cleanup: {e}")

    def _initialize_state(self, ticket_id: str) -> TestExecutionState:
        """Initialize execution state"""
        from datetime import datetime

        # Generate single timestamp for all artifacts (report, script, video)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            'timestamp': timestamp,  # Single timestamp for all artifacts
            'config': self.config
        }
        return state

    def _parse_ticket_llm(self, state: TestExecutionState) -> Dict[str, Any]:
        """Parse ticket using JiraAgent"""
        print("\n[1/6] Parsing ticket...", end="", flush=True)

        ticket_data = self.jira_agent.parse_ticket(state['ticket_id'])

        state['ticket_title'] = ticket_data.get('title', '')
        state['module'] = ticket_data.get('module', '')
        state['total_steps'] = len(ticket_data.get('steps', []))
        state['agent_chain'].append('JiraAgent')

        print(f" [OK]")
        print(f"Ticket: {ticket_data.get('title', 'N/A')}")
        print(f"Module: {state['module']} | Steps: {state['total_steps']}")

        return ticket_data

    def _init_browser(self, playwright) -> Browser:
        """Initialize browser"""
        print("\n[2/6] Initializing browser...", end="", flush=True)

        browser_type = self.config['browser']
        headless = self.config['execution']['headless']

        if browser_type == 'edge':
            browser = playwright.chromium.launch(
                headless=headless,
                channel='msedge',
                args=['--start-maximized']
            )
        else:
            browser = playwright.chromium.launch(
                headless=headless,
                args=['--start-maximized']
            )

        print(f" [OK]")
        return browser

    def _capture_initial_context(self, state: TestExecutionState, page: Page):
        """Capture initial context"""
        print("\n[3/6] Capturing initial context...", end="", flush=True)

        page.goto(self.config['web_url'])
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(2000)

        # Capture context
        context = self.context_agent.capture_context(page)
        self.context_agent.update_history(state, context)
        state['agent_chain'].append('ContextAgent')

        print(f" [OK]")

    def _login(self, state: TestExecutionState, page: Page):
        """Login to application"""
        print("\n[4/6] Logging in...", end="", flush=True)

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

            print(f" [OK]")

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

            # Print clean step progress (truncate long text)
            step_display = step_text if len(step_text) <= 70 else step_text[:67] + "..."
            print(f"\n  Step {step_number}/{len(steps)}: {step_display}", end="", flush=True)

            # Skip "Login" step if already logged in
            if step_text.lower().strip() in ['login', 'log in', 'sign in', 'signin']:
                logger.debug(f"Detected login step. logged_in flag: {state.get('logged_in', False)}")
                if state.get('logged_in', False):
                    print(f" [SKIPPED]")

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
                        logger.debug(f"Text verification step detected, using L3 Vision Agent")

                        # FIX: Wait for success messages to stabilize before verification
                        # Success messages often fade/disappear quickly, causing flaky tests
                        text_verify_wait = self.config.get('wait_times', {}).get('before_text_verification', 2000)
                        if text_verify_wait > 0:
                            logger.debug(f"Waiting {text_verify_wait}ms for message to stabilize")
                            page.wait_for_timeout(text_verify_wait)

                        # Use screenshot from previous step if available
                        prev_screenshot = state.get('last_screenshot')
                        l3_result = self.selector_agent_l3.verify_text(page, expected_text, step_text, prev_screenshot)
                        state['agent_chain'].append('SelectorAgent_L3')

                        success = l3_result['found']
                        selector = f"Text verification: '{expected_text}'"
                        confidence = l3_result['confidence']
                        agent_used = 'L3'

                        logger.debug(f"L3 vision result: {l3_result}")

                        # Record result
                        step_result = {
                            'step_number': step_number,
                            'step_text': step_text,
                            'selector': selector,
                            'confidence': confidence,
                            'agent_used': agent_used,
                            'action_type': 'verify_text',
                            'verified_text': l3_result.get('actual_text', expected_text),  # Actual text found by L3
                            'status': 'PASSED' if success else 'FAILED',
                            'context': context,
                            'l3_result': l3_result
                        }

                        state['step_results'].append(step_result)

                        if success:
                            print(f" [OK] (L3)")
                        else:
                            print(f" [FAILED] (L3)")
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

                    # Check if L1 selector is valid (found in visible elements)
                    selector_validated = l1_result.get('selector_validated', True)  # Default to True if not set

                    if l1_result['selector'] and l1_result['confidence'] >= 0.70:
                        # Use L1 result if confidence is good
                        # Only fall back to L2 if explicitly marked as NOT validated AND confidence is borderline
                        if not selector_validated and l1_result['confidence'] < 0.75:
                            # Selector not validated AND low-ish confidence → Try L2
                            print(f"  [L1 selector not validated, trying L2...]")

                            l2_result = self.selector_agent_l2.discover_selector(page, step_text, context)
                            state['agent_chain'].append('SelectorAgent_L2')

                            if l2_result['selector'] and l2_result['confidence'] >= 0.70:
                                selector = l2_result['selector']
                                confidence = l2_result['confidence']
                                agent_used = 'L2'
                                print(f"  [L2] {selector} (conf: {confidence:.2f})")
                            else:
                                # L2 failed, fall back to L1 anyway
                                print(f"  [L2 failed, using L1 anyway]")
                                selector = l1_result['selector']
                                confidence = l1_result['confidence']
                                agent_used = 'L1'
                                print(f"  [L1] {selector} (conf: {confidence:.2f})")
                        else:
                            # L1 confidence good, use it (even if not validated - validation is not 100% reliable)
                            selector = l1_result['selector']
                            confidence = l1_result['confidence']
                            agent_used = 'L1'
                            logger.debug(f"[L1] {selector} (conf: {confidence:.2f})")

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
                    print(f" [OK] ({agent_used})")

                    # Store to learning for future use
                    self.learning_agent.store_learned_selector(
                        step_text, selector, context, confidence, source='runtime', verified=True
                    )
                else:
                    print(f" [FAILED]")
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
                print(f" [FAILED] {str(e)[:50]}")

                if self.config.get('execution', {}).get('failure_handling', {}).get('fail_fast', False):
                    print(f"\n[FAIL-FAST] Stopping execution")
                    break

        print("-" * 80)

    def _extract_dynamic_value(self, step_text: str) -> str:
        """
        Extract dynamic value from step text for selector substitution.

        Examples:
          "Choose Type 4" → "Type 4"
          "Edit default_testobject_02" → "default_testobject_02"
          "Select Measurement10" → "Measurement10"

        Args:
            step_text: The step description

        Returns:
            Extracted dynamic value or empty string
        """
        import re

        step_lower = step_text.lower()

        # Pattern 1: After action keywords (choose/select/edit/click/delete)
        # "Choose Type 4" → "Type 4"
        match = re.search(r'(?:choose|select|edit|click on|delete|remove)\s+(.+?)(?:\s+in|\s+from|\s+dropdown|\s+menu|$)', step_lower, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Find the actual case-sensitive value from original text
            try:
                start_idx = step_text.lower().index(value)
                return step_text[start_idx:start_idx + len(value)]
            except:
                return value

        # Pattern 2: Quoted text 'Type 4' or "Type 4"
        match = re.search(r'["\']([^"\']+)["\']', step_text)
        if match:
            return match.group(1)

        # Pattern 3: Names with underscores and numbers (default_testobject_01)
        match = re.search(r'\b([a-zA-Z_]+[a-zA-Z0-9_]*\d+[a-zA-Z0-9_]*)\b', step_text)
        if match:
            return match.group(1)

        # Pattern 4: Capitalized words followed by numbers (Type 4, Measurement10)
        match = re.search(r'\b([A-Z][a-z]+\s*\d+)\b', step_text)
        if match:
            return match.group(1)

        return ""

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
            # Get timeout from config with fallback to 30000ms (30 seconds) for slow-loading elements
            timeout_ms = self.config.get('performance', {}).get('element_wait_timeout', 30000)
            page.wait_for_selector(selector, timeout=timeout_ms)

            # Check if this is a save/submit action
            step_lower = step_text.lower()
            is_save_action = any(word in step_lower for word in ['save', 'submit', 'create', 'update', 'delete'])

            if action_type == 'click':
                page.click(selector, timeout=timeout_ms)
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
                page.click(selector, timeout=timeout_ms)
                page.wait_for_timeout(self.config['wait_times']['after_navigation'])

                # Smart wait for table data loading after navigation
                # Wait for any data-attribute elements to appear (indicates table loaded)
                try:
                    logger.info("Waiting for table data to load...")
                    page.wait_for_selector('[data-attribute]', timeout=10000, state='visible')
                    logger.info("Table data loaded successfully")
                except Exception as e:
                    logger.warning(f"Table data may not have loaded: {e}")
                    # Continue anyway - maybe no table on this page

            # Capture screenshot after action for next step's verification
            if state is not None:
                import base64
                screenshot_bytes = page.screenshot()
                state['last_screenshot'] = base64.b64encode(screenshot_bytes).decode('utf-8')
                logger.debug(f"Captured post-action screenshot ({len(screenshot_bytes)} bytes)")

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
        print("\n[6/6] Generating artifacts...", end="", flush=True)

        execution_time = time.time() - state['execution_start_time']

        passed_steps = sum(1 for r in state['step_results'] if r.get('status') == 'PASSED')
        failed_steps = sum(1 for r in state['step_results'] if r.get('status') == 'FAILED')

        summary = {
            "ticket_id": state['ticket_id'],
            "module": state['module'],  # FIX: Add module to results for feedback collection
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
                config=self.config,
                timestamp=state['timestamp']  # Pass shared timestamp
            )

            summary['report_path'] = report_path
            logger.debug(f"HTML Report: {report_path}")
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
                    config=self.config,
                    timestamp=state['timestamp']  # Pass shared timestamp
                )
                summary['script_path'] = script_path
                logger.debug(f"Playwright Script: {script_path}")

                # Generate conftest.py and README (one-time generation)
                try:
                    generate_pytest_config(self.config)
                    generate_readme(self.config)
                except Exception as e:
                    logger.warning(f"Config/README generation skipped: {e}")
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
            logger.debug(f"Context Trace: {context_trace_path}")
        except Exception as e:
            logger.error(f"Context trace save failed: {e}")

        print(f" [OK]")

        # Clean summary output
        print(f"\n{'='*80}")
        status = summary['overall_status']
        if status == 'PASSED':
            print(f"✓ TEST PASSED")
        else:
            print(f"✗ TEST FAILED")
        print(f"{'='*80}")

        print(f"Steps: {summary['total_steps']} total | {summary['passed_steps']} passed | {summary['failed_steps']} failed")
        print(f"Time: {summary['execution_time']}")

        # Agent usage stats
        agent_counts = {}
        for agent in state['agent_chain']:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        l1_count = agent_counts.get('SelectorAgent_L1', 0)
        l2_count = agent_counts.get('SelectorAgent_L2', 0)
        l3_count = agent_counts.get('SelectorAgent_L3', 0)

        print(f"Agents: L1:{l1_count} | L2:{l2_count} | L3:{l3_count}")
        print(f"\nReport: {summary.get('report_path', 'Not generated')}")
        print(f"{'='*80}")

        return summary


def collect_and_process_feedback(
    ticket_id: str,
    results: Dict[str, Any],
    config: Dict[str, Any],
    assistant: 'PLCDTestingAssistantSeq'
) -> bool:
    """
    Collect natural language feedback from tester and process it.

    Args:
        ticket_id: Test ticket ID
        results: Test execution results
        config: Configuration dictionary
        assistant: PLCDTestingAssistantSeq instance

    Returns:
        True if feedback was collected and processed, False otherwise
    """
    from agents.feedback_parser import FeedbackParserAgent
    from agents.insight_enhancer import InsightEnhancerAgent
    from insight_storage import InsightStorage

    try:
        print("\n" + "="*80)
        print("FEEDBACK COLLECTION")
        print("="*80)

        # Show HTML report location
        html_report = results.get('html_report_path', 'Reports/*.html')
        print(f"\nHTML Report: {html_report}")
        print("\nPlease review the HTML report and provide feedback.")
        print("\nExamples of feedback:")
        print("  - 'Step 5 is false positive, should use [data-editnodebtn='default_UUT_01']'")
        print("  - 'Step 6 failed, correct selector is [data-checkuniquename=\"attribut1\"]'")
        print("  - 'Step 3 has timing issue, needs longer wait'")
        print("  - 'All steps correct, test result is accurate'")
        print("\nType your feedback below (press Enter twice when done - once after text, once on empty line):")
        print("-" * 80)

        # Collect multi-line feedback
        feedback_lines = []
        while True:
            try:
                line = input()
                if line.strip() == '':
                    break
                feedback_lines.append(line)
            except EOFError:
                break

        feedback_text = '\n'.join(feedback_lines).strip()

        # If no feedback provided, exit
        if not feedback_text:
            print("\n[INFO] No feedback provided. Skipping feedback processing.")
            return False

        print("\n" + "-"*80)
        print("Processing feedback...")

        # Initialize agents and storage
        parser = FeedbackParserAgent(config)
        enhancer = InsightEnhancerAgent(config)
        storage = InsightStorage()  # Uses default path: ./insights

        # Build test context for parsing
        # Note: We use a large number for total_steps to avoid validation issues
        # when tester provides feedback for steps that weren't executed
        test_context = {
            'ticket_id': ticket_id,
            'steps': [{'text': f'Step {i}'} for i in range(1, 100)],  # Dummy steps for validation
            'results': {
                'step_results': results.get('step_results', []),
                'overall_status': results.get('overall_status', 'UNKNOWN')
            }
        }

        # Parse natural language feedback
        parsed_feedback = parser.parse_feedback(feedback_text, test_context)

        if not parsed_feedback or 'steps' not in parsed_feedback:
            print("\n[WARNING] Could not parse feedback. Please check format.")
            return False

        # Validate parsed feedback
        if not parser.validate_parsed_feedback(parsed_feedback, test_context):
            print("\n[WARNING] Parsed feedback validation failed.")
            return False

        print(f"✓ Parsed feedback for {len(parsed_feedback.get('steps', {}))} steps")

        # Get actual step texts from results for accurate matching
        actual_step_results = results.get('step_results', [])

        # Process each step's feedback
        feedback_steps = parsed_feedback.get('steps', {})
        insights_saved = 0

        for step_key, step_feedback in feedback_steps.items():
            # Extract step number
            step_num = int(step_key.replace('step_', ''))

            # Get step text from ACTUAL test execution results (not dummy steps)
            step_text = ""
            for step_result in actual_step_results:
                if step_result.get('step_number') == step_num or step_result.get('step') == step_num:
                    step_text = step_result.get('step_text', step_result.get('text', ''))
                    break

            # Fallback if step not found in results (e.g., step never executed)
            if not step_text:
                step_text = f"Step {step_num}"  # Use generic text as last resort

            # Build insight structure (using correct field names for LLM search compatibility)
            feedback_type = step_feedback.get('feedback_type', 'failed')
            # FIX: Get module from results, NOT from sequential_context (which was empty)
            module = results.get('module', '')

            # Infer action type from step text
            step_lower = step_text.lower()
            if any(word in step_lower for word in ['edit', 'modify', 'update']):
                action_type = 'edit'
            elif any(word in step_lower for word in ['delete', 'remove']):
                action_type = 'delete'
            elif any(word in step_lower for word in ['save', 'submit', 'confirm']):
                action_type = 'save'
            elif any(word in step_lower for word in ['click', 'select', 'choose', 'expand']):
                action_type = 'click'
            elif any(word in step_lower for word in ['verify', 'check', 'assert']):
                action_type = 'verify'
            else:
                action_type = 'interact'

            insight = {
                # Fields required by search_pending_insights_llm() - DO NOT CHANGE THESE NAMES
                'step': step_text,                                          # Used by LLM search
                'selector': step_feedback.get('correct_selector', ''),     # Used by LLM search
                'confidence': 0.95,                                        # High confidence for tester feedback
                'category': feedback_type,                                 # corrections, failed, etc.
                'module': module,                                          # FIX: Add module at top level for insight_storage.py

                # Context required by search function and batch embedding
                'context': {
                    'module': module,
                    'action_type': action_type,
                    'sequential_context': {
                        'previous_steps': [],
                        'last_successful_action': None,
                        'last_selector_used': None,
                        'page_state_before_failure': f"After step {step_num - 1}" if step_num > 1 else "Initial state",
                        'current_module': module
                    }
                },

                # Metadata for tracking and debugging
                'metadata': {
                    'ticket_id': ticket_id,
                    'step_number': step_num,
                    'timestamp': datetime.now().isoformat(),
                    'issue_description': step_feedback.get('issue_description', ''),
                    'reasoning': step_feedback.get('reasoning', ''),
                    'browser': config.get('browser', 'chromium') if isinstance(config.get('browser'), str) else config.get('browser', {}).get('name', 'chromium'),
                    'tester_id': 'manual_feedback',
                    'source': 'tester_feedback',
                    'feedback_type': feedback_type  # Keep for backward compatibility
                }
            }

            # Generate embedding for the step text (for precise matching)
            try:
                from config_loader import get_azure_client
                azure_client = get_azure_client(config)
                embedding_model = config['azure_openai']['models']['embedding']

                # Create combined text for embedding (step + module context)
                embed_text = f"{step_text} {module}".strip()

                response = azure_client.embeddings.create(
                    input=embed_text,
                    model=embedding_model
                )
                insight['embedding'] = response.data[0].embedding
                logger.info(f"Generated embedding for Step {step_num} ({len(insight['embedding'])} dims)")
            except Exception as e:
                logger.warning(f"Failed to generate embedding for Step {step_num}: {e}")
                # Continue without embedding - will fall back to LLM search
                insight['embedding'] = None

            # Save insight to appropriate category
            try:
                filepath = storage.save_insight_by_category(
                    insight=insight,
                    ticket_id=ticket_id,
                    step_number=step_num
                )
                print(f"✓ Saved insight for Step {step_num}: {step_feedback.get('feedback_type')}")
                insights_saved += 1
            except Exception as e:
                logger.error(f"Failed to save insight for step {step_num}: {e}")
                print(f"✗ Failed to save insight for Step {step_num}")

        print(f"\n[OK] Processed {insights_saved} insights from feedback")
        print("="*80)

        return insights_saved > 0

    except KeyboardInterrupt:
        print("\n\n[INFO] Feedback collection cancelled by user.")
        return False
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to collect feedback: {e}")
        return False


def main():
    """Main entry point with continuous feedback loop"""

    if len(sys.argv) < 2:
        print("Usage: python plcd_taseq.py <TICKET_ID> [--no-feedback]")
        print("Example: python plcd_taseq.py RBPLCD-8835")
        print("         python plcd_taseq.py RBPLCD-8835 --no-feedback")
        sys.exit(1)

    ticket_id = sys.argv[1]
    no_feedback = '--no-feedback' in sys.argv

    try:
        # Load configuration
        config = load_config()

        # Initialize assistant
        assistant = PLCDTestingAssistantSeq(config)

        # Continuous feedback loop
        iteration = 1
        final_results = None

        while True:
            print("\n" + "="*80)
            if iteration == 1:
                print(f"Test Execution - Iteration {iteration}")
            else:
                print(f"Test Retry - Iteration {iteration} (with learned insights)")
            print("="*80 + "\n")

            # Execute test
            results = assistant.execute_test(ticket_id)
            final_results = results

            # Collect feedback if enabled (ALWAYS ask, even if passed)
            if not no_feedback:
                print("\n" + "="*80)
                print("FEEDBACK COLLECTION")
                print("="*80)
                print("\nTest completed. Please review the results and provide feedback.")
                print("You can correct false positives, failed steps, or confirm accuracy.\n")

                feedback_collected = collect_and_process_feedback(
                    ticket_id=ticket_id,
                    results=results,
                    config=config,
                    assistant=assistant
                )

                # Ask user if they want to retry
                if feedback_collected:
                    print("\n" + "="*80)
                    try:
                        user_choice = input("\nRetry test with learned insights? (y/n/stop): ").strip().lower()
                    except EOFError:
                        print("EOF detected, stopping.")
                        break

                    if user_choice in ['n', 'no', 'stop', 'exit', 'quit']:
                        print("\n[INFO] Feedback loop stopped by user.")
                        break
                    elif user_choice in ['y', 'yes']:
                        iteration += 1
                        continue
                    else:
                        print("\n[INFO] Invalid input. Stopping feedback loop.")
                        break
                else:
                    # No feedback provided
                    print("\n[INFO] No feedback provided. Exiting feedback loop.")
                    break
            else:
                # No feedback mode
                break

        # Final summary
        print("\n" + "="*80)
        print("FINAL RESULT")
        print("="*80)
        print(f"Total iterations: {iteration}")
        print(f"Final status: {final_results['overall_status']}")
        print("="*80)

        # Exit with appropriate code
        sys.exit(0 if final_results['overall_status'] == 'PASSED' else 1)

    except KeyboardInterrupt:
        print("\n\n[INFO] Test execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# NEW: Vector Search Helper Functions (Learned Insights)
# ============================================================================

def search_learned_insights(chroma_client, azure_client, config: Dict[str, Any],
                           step_text: str, current_module: str = '') -> List[Dict[str, Any]]:
    """
    Search learned_insights_collection for semantically similar insights.

    Args:
        chroma_client: ChromaDB client
        azure_client: Azure OpenAI client
        config: Configuration dictionary
        step_text: Step description to search for
        current_module: Current module context

    Returns:
        List of matching insights with selectors and confidence scores
    """
    try:
        # Get collection
        collection_name = config['vector_database']['collections']['learned_insights']
        collection = chroma_client.get_collection(name=collection_name)

        # Get config values
        vector_search_config = config.get('vector_search', {})
        max_results = vector_search_config.get('learned_max_results', 3)
        similarity_threshold = vector_search_config.get('similarity_threshold', 0.85)
        confidence_boost = vector_search_config.get('learned_confidence_boost', 0.2)

        # Build query text with module context
        query_text = f"{step_text} {current_module}".strip()

        # Generate embedding
        embedding_model = config['azure_openai']['models']['embedding']
        response = azure_client.embeddings.create(
            input=query_text,
            model=embedding_model
        )
        query_embedding = response.data[0].embedding

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results
        )

        if not results['ids'] or len(results['ids'][0]) == 0:
            logger.debug("No learned insights found")
            return []

        # Process results
        matches = []
        for idx in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][idx]
            distance = results['distances'][0][idx]

            # Convert distance to similarity
            similarity = 1.0 - distance

            if similarity >= similarity_threshold:
                # Apply confidence boost
                boosted_confidence = min(similarity + confidence_boost, 1.0)

                matches.append({
                    'selector': metadata.get('selector', ''),
                    'confidence': boosted_confidence,
                    'similarity': similarity,
                    'source': 'learned_insights',
                    'category': metadata.get('category', 'unknown'),
                    'module': metadata.get('module', ''),
                    'ticket_id': metadata.get('ticket_id', '')
                })

        logger.debug(f"Found {len(matches)} learned insights (similarity >= {similarity_threshold})")
        return matches

    except Exception as e:
        logger.error(f"Failed to search learned insights: {e}")
        return []


def search_pending_insights_llm(azure_client, config: Dict[str, Any],
                                step_text: str, current_module: str = '') -> List[Dict[str, Any]]:
    """
    Search pending/ folder for similar insights using LLM comparison.

    Args:
        azure_client: Azure OpenAI client
        config: Configuration dictionary
        step_text: Step description to search for
        current_module: Current module context

    Returns:
        List of matching insights with selectors and confidence scores
    """
    try:
        from pathlib import Path
        import json

        # Get config values
        vector_search_config = config.get('vector_search', {})
        pending_folder = Path(vector_search_config.get('pending_folder', 'insights/pending'))
        max_results = vector_search_config.get('pending_max_results', 3)
        similarity_threshold = vector_search_config.get('pending_threshold', 0.80)
        confidence_boost = vector_search_config.get('pending_confidence_boost', 0.15)
        llm_config = vector_search_config.get('llm_comparison', {})

        if not llm_config.get('enabled', True):
            return []

        if not pending_folder.exists():
            return []

        # Load all pending insights
        pending_insights = []
        for json_file in pending_folder.glob("**/*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    insight = json.load(f)
                    pending_insights.append(insight)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

        if not pending_insights:
            return []

        # Build comparison prompt
        comparison_list = []
        for idx, insight in enumerate(pending_insights, start=1):
            step = insight.get('step', '')
            selector = insight.get('selector', '')
            module = insight.get('context', {}).get('module', '') if isinstance(insight.get('context'), dict) else ''
            comparison_list.append(f"{idx}. \"{step}\" (Module: {module}) → {selector}")

        comparison_text = "\n".join(comparison_list)

        prompt = f"""Current step: "{step_text}"
Current module: "{current_module}"

Compare with these learned steps and return the best matches:
{comparison_text}

Return JSON with top {max_results} matches (or fewer if similarity < {similarity_threshold}):
{{
  "matches": [
    {{"index": 1, "similarity": 0.92, "reason": "Both are edit actions"}},
    {{"index": 3, "similarity": 0.88, "reason": "Similar navigation step"}}
  ]
}}

If no good matches (all < {similarity_threshold}), return: {{"matches": []}}
"""

        # Call LLM
        chat_model = llm_config.get('model', 'gpt-4o')
        response = azure_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a semantic similarity expert. Compare test steps and return matching indices with similarity scores."},
                {"role": "user", "content": prompt}
            ],
            temperature=llm_config.get('temperature', 0.1),
            max_tokens=llm_config.get('max_tokens', 500)
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if not json_match:
            logger.warning("LLM response not valid JSON")
            return []

        result = json.loads(json_match.group())
        llm_matches = result.get('matches', [])

        # Build final matches
        matches = []
        for match in llm_matches:
            idx = match.get('index', 0) - 1  # Convert to 0-based
            similarity = match.get('similarity', 0.0)

            if idx < 0 or idx >= len(pending_insights):
                continue

            if similarity >= similarity_threshold:
                insight = pending_insights[idx]
                boosted_confidence = min(similarity + confidence_boost, 1.0)

                matches.append({
                    'selector': insight.get('selector', ''),
                    'confidence': boosted_confidence,
                    'similarity': similarity,
                    'source': 'pending_insights',
                    'category': insight.get('category', 'unknown'),
                    'module': insight.get('context', {}).get('module', ''),
                    'reason': match.get('reason', '')
                })

        logger.debug(f"Found {len(matches)} pending insights via LLM comparison")
        return matches

    except Exception as e:
        logger.error(f"Failed to search pending insights: {e}")
        return []


def search_pending_insights_embeddings(azure_client, config: Dict[str, Any],
                                       step_text: str, current_module: str = '') -> List[Dict[str, Any]]:
    """
    Search ALL insight folders for similar insights using embedding-based cosine similarity.
    Searches: corrections, verified, pending, flaky, bugs, negative_tests.
    Uses timestamp tie-breaking for insights with similar semantic scores.

    Args:
        azure_client: Azure OpenAI client
        config: Configuration dictionary
        step_text: Step description to search for
        current_module: Current module context

    Returns:
        List of matching insights with selectors and confidence scores (best match first)
    """
    try:
        from pathlib import Path
        import json
        import numpy as np

        # Get config values
        vector_search_config = config.get('vector_search', {})
        insights_base = Path(vector_search_config.get('pending_folder', 'insights/pending')).parent
        max_results = vector_search_config.get('pending_max_results', 3)
        similarity_threshold = vector_search_config.get('pending_threshold', 0.75)
        confidence_boost = vector_search_config.get('pending_confidence_boost', 0.15)

        # Define all insight folders in priority order
        insight_folders = [
            insights_base / 'corrections',      # False positives (highest priority)
            insights_base / 'verified',         # True positives
            insights_base / 'pending',          # Regular failures
            insights_base / 'flaky',            # Timing issues
            insights_base / 'negative_tests',   # Expected failures
            insights_base / 'bugs'              # Application bugs
        ]

        # Generate query embedding
        embedding_model = config['azure_openai']['models']['embedding']
        query_text = f"{step_text} {current_module}".strip()

        response = azure_client.embeddings.create(
            input=query_text,
            model=embedding_model
        )
        query_embedding = np.array(response.data[0].embedding)

        # Load all insights with embeddings from ALL folders
        candidates = []
        for folder in insight_folders:
            if not folder.exists():
                continue

            for json_file in folder.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        insight = json.load(f)

                        # Check if insight has embedding
                        if 'embedding' not in insight or insight['embedding'] is None:
                            logger.debug(f"Skipping {json_file.name} - no embedding")
                            continue

                        stored_embedding = np.array(insight['embedding'])

                        # Calculate cosine similarity
                        dot_product = np.dot(query_embedding, stored_embedding)
                        norm_query = np.linalg.norm(query_embedding)
                        norm_stored = np.linalg.norm(stored_embedding)
                        similarity = dot_product / (norm_query * norm_stored)

                        # Get timestamp for tie-breaking
                        timestamp = insight.get('metadata', {}).get('timestamp', '')
                        if not timestamp:
                            timestamp = insight.get('storage_metadata', {}).get('saved_at', '')

                        candidates.append({
                            'insight': insight,
                            'similarity': float(similarity),
                            'timestamp': timestamp,
                            'file': json_file.name,
                            'folder': folder.name
                        })

                except Exception as e:
                    logger.warning(f"Failed to process {json_file}: {e}")
                    continue

        # Filter by threshold
        filtered = [c for c in candidates if c['similarity'] >= similarity_threshold]

        # Sort by similarity (descending), then by timestamp (descending) for tie-breaking
        # This ensures newest insights win when similarities are very close
        filtered.sort(key=lambda x: (x['similarity'], x['timestamp']), reverse=True)

        logger.debug(f"Found {len(filtered)} insights >= {similarity_threshold} similarity threshold")

        # Apply tie-breaking: If top results have very similar scores (within 0.02), prefer newest
        if len(filtered) > 1:
            best_similarity = filtered[0]['similarity']
            tied_candidates = [c for c in filtered if abs(c['similarity'] - best_similarity) < 0.02]

            if len(tied_candidates) > 1:
                # Multiple insights with very similar scores - use newest
                tied_candidates.sort(key=lambda x: x['timestamp'], reverse=True)
                logger.debug(f"Tie-breaking: {len(tied_candidates)} insights within 0.02 similarity")
                logger.debug(f"Selected newest: {tied_candidates[0]['file']} (timestamp: {tied_candidates[0]['timestamp']})")

                # Rebuild filtered list with tie-broken order
                remaining = [c for c in filtered if c not in tied_candidates]
                filtered = tied_candidates + remaining

        # Return top N matches
        matches = []
        for candidate in filtered[:max_results]:
            insight = candidate['insight']
            similarity = candidate['similarity']
            boosted_confidence = min(similarity + confidence_boost, 1.0)

            matches.append({
                'selector': insight.get('selector', ''),
                'confidence': boosted_confidence,
                'similarity': similarity,
                'source': 'pending_insights_embeddings',
                'category': insight.get('category', 'unknown'),
                'module': insight.get('context', {}).get('module', ''),
                'file': candidate['file'],
                'folder': candidate['folder'],
                'timestamp': candidate['timestamp'],
                'metadata': insight.get('metadata', {})  # Include metadata for dynamic substitution
            })

        logger.debug(f"Returning {len(matches)} best matches (similarity >= {similarity_threshold})")
        if matches:
            logger.debug(f"Best match: {matches[0]['file']} (similarity: {matches[0]['similarity']:.3f}, timestamp: {matches[0]['timestamp']})")

        return matches

    except Exception as e:
        logger.error(f"Failed to search pending insights with embeddings: {e}")
        return []


if __name__ == "__main__":
    main()
