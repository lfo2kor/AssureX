"""
PLCD Testing Assistant - Agent 2: DOM Discovery (L2)
Analyzes live page DOM when Agent 1 has medium confidence (0.60-0.75)
"""

import logging
from typing import Dict, List, Any, Optional
from playwright.sync_api import Page
import numpy as np

from config_loader import load_config, get_azure_client, get_embedding_model


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/agent2_dom_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Agent2DOMDiscovery:
    """
    Agent 2: Live DOM Analysis and Selector Discovery

    Activated when Agent 1 confidence is between 0.60-0.75
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Agent 2

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.azure_client = get_azure_client(config)
        self.embedding_model = get_embedding_model(config)
        self.agent_config = config.get('agent2_dom_discovery', {})

        logger.info("Agent 2: DOM Discovery initialized")


    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.azure_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding


    def extract_dom_elements(self, page: Page) -> List[Dict]:
        """
        Extract interactive elements from live page

        Args:
            page: Playwright page object

        Returns:
            List of element dictionaries
        """
        logger.info("Extracting DOM elements from page...")

        # JavaScript to extract elements
        dom_elements = page.evaluate("""
            () => {
                const elements = [];
                const selectors = document.querySelectorAll(
                    'button, input, select, a, [role="button"], [data-test], [data-testid], span, div'
                );

                selectors.forEach((el, idx) => {
                    const attrs = {};

                    // Extract data-* attributes (highest priority)
                    for (let attr of el.attributes) {
                        if (attr.name.startsWith('data-')) {
                            attrs[attr.name] = attr.value;
                        }
                    }

                    // ARIA attributes
                    if (el.getAttribute('aria-label')) attrs['aria-label'] = el.getAttribute('aria-label');
                    if (el.getAttribute('role')) attrs['role'] = el.getAttribute('role');

                    // Other stable attributes
                    if (el.id && !el.id.match(/^[0-9]+$/)) attrs['id'] = el.id;
                    if (el.name) attrs['name'] = el.name;

                    // Get visible text
                    const text = el.innerText?.substring(0, 100) || el.textContent?.substring(0, 100) || '';

                    // Check visibility
                    const rect = el.getBoundingClientRect();
                    const visible = rect.width > 0 && rect.height > 0 &&
                                   window.getComputedStyle(el).visibility !== 'hidden' &&
                                   window.getComputedStyle(el).display !== 'none';

                    if (visible && Object.keys(attrs).length > 0) {
                        elements.push({
                            index: idx,
                            tagName: el.tagName.toLowerCase(),
                            text: text.trim(),
                            attributes: attrs,
                            visible: visible
                        });
                    }
                });

                return elements;
            }
        """)

        logger.info(f"Extracted {len(dom_elements)} visible elements with stable attributes")
        return dom_elements


    def build_selector(self, element: Dict) -> str:
        """
        Build stable CSS selector from element

        Priority:
        1. data-* attributes
        2. aria-* attributes
        3. id
        4. name
        5. tag + text

        Args:
            element: Element dictionary

        Returns:
            CSS selector string
        """
        attrs = element.get('attributes', {})

        # Priority 1: data-* attributes
        for attr_name, attr_value in attrs.items():
            if attr_name.startswith('data-'):
                return f"[{attr_name}='{attr_value}']"

        # Priority 2: ARIA
        if 'aria-label' in attrs:
            return f"[aria-label='{attrs['aria-label']}']"
        if 'role' in attrs:
            tag = element.get('tagName', '*')
            return f"{tag}[role='{attrs['role']}']"

        # Priority 3: ID
        if 'id' in attrs:
            return f"#{attrs['id']}"

        # Priority 4: Name
        if 'name' in attrs:
            tag = element.get('tagName', '*')
            return f"{tag}[name='{attrs['name']}']"

        # Fallback: tag with text
        tag = element.get('tagName', 'button')
        text = element.get('text', '')[:30]
        if text:
            return f"{tag}:has-text('{text}')"

        return f"{tag}"


    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


    def discover_selector(
        self,
        page: Page,
        step_text: str,
        agent1_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Discover selector from live page DOM

        Args:
            page: Playwright page object
            step_text: Test step description
            agent1_result: Optional result from Agent 1

        Returns:
            Dictionary containing selector result
        """
        logger.info("=" * 80)
        logger.info("Agent 2: DOM Discovery")
        logger.info(f"Step: {step_text}")

        # Extract DOM elements
        dom_elements = self.extract_dom_elements(page)

        if not dom_elements:
            logger.warning("No DOM elements found")
            return {"selector_result": None, "candidates": []}

        # Generate embedding for step
        step_embedding = self.generate_embedding(step_text)

        # Generate embeddings for each element
        logger.info(f"Generating embeddings for {len(dom_elements)} elements...")
        element_embeddings = []
        element_texts = []

        for elem in dom_elements:
            # Create composite text (similar to Agent 1 strategy)
            tag = elem.get('tagName', '')
            text = elem.get('text', '')
            attrs_str = ' '.join(elem.get('attributes', {}).values())
            composite = f"{tag} {text} {attrs_str}".strip()

            element_texts.append(composite)

        # Batch generate embeddings
        if element_texts:
            response = self.azure_client.embeddings.create(
                input=element_texts,
                model=self.embedding_model
            )
            element_embeddings = [data.embedding for data in response.data]

        # Find best semantic match
        best_match_idx = -1
        best_similarity = 0.0

        for idx, elem_embedding in enumerate(element_embeddings):
            similarity = self.cosine_similarity(step_embedding, elem_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        if best_match_idx == -1:
            logger.warning("No matching element found")
            return {"selector_result": None, "candidates": []}

        # Build selector
        best_element = dom_elements[best_match_idx]
        selector = self.build_selector(best_element)

        # Calculate confidence with boosting
        confidence = best_similarity

        # Boost confidence if element has data-* attributes
        attrs = best_element.get('attributes', {})
        if any(attr.startswith('data-') for attr in attrs):
            confidence += self.agent_config.get('confidence_boost', {}).get('has_data_attr', 0.15)

        if 'aria-label' in attrs or 'role' in attrs:
            confidence += self.agent_config.get('confidence_boost', {}).get('has_aria', 0.10)

        confidence = min(confidence, 1.0)  # Cap at 1.0

        # Validate selector on page
        try:
            page.wait_for_selector(selector, timeout=2000)
            is_valid = True
        except:
            is_valid = False
            confidence *= 0.7  # Penalize if not found

        result = {
            "selector": selector,
            "confidence": confidence,
            "agent_used": "L2",
            "metadata": {
                "element": best_element,
                "similarity": best_similarity,
                "validated": is_valid,
                "dom_elements_analyzed": len(dom_elements)
            }
        }

        logger.info(f"Best match: {selector}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Validated: {is_valid}")
        logger.info("=" * 80)

        return {
            "selector_result": result,
            "candidates": [result]
        }


if __name__ == "__main__":
    print("Agent 2: DOM Discovery")
    print("This agent requires a live Playwright page instance.")
    print("Use within plcd_ta.py for testing.")
