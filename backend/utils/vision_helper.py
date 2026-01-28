"""
Azure OpenAI GPT-4o Vision API wrapper.

Provides interface for calling GPT-4o vision model with screenshot images,
including retry logic for API errors and rate limits.
"""

import base64
import json
import time
import logging
from typing import Dict, Any, Optional
from openai import AzureOpenAI, RateLimitError, APIError, APITimeoutError


class AzureVisionClient:
    """
    Client for Azure OpenAI GPT-4o vision API.

    Handles:
    - Image encoding to base64
    - API calls with proper message formatting
    - JSON response parsing
    - Retry logic for rate limits and timeouts
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize Azure OpenAI vision client.

        Args:
            config: Configuration dictionary with azure_openai settings
            logger: Logger instance for logging API calls

        Raises:
            KeyError: If required configuration fields are missing
        """
        self.config = config
        self.logger = logger
        self.max_retries = 3

        # Initialize Azure OpenAI client
        azure_config = config['azure_openai']
        self.client = AzureOpenAI(
            api_key=azure_config['api_key'],
            api_version=azure_config['api_version'],
            azure_endpoint=azure_config['endpoint']
        )
        self.deployment = azure_config['deployment_gpt4o']

        self.logger.info("Azure Vision Client initialized")
        self.logger.debug(f"Using deployment: {self.deployment}")

    def encode_image(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string.

        Args:
            image_bytes: Raw image data

        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_bytes).decode('utf-8')

    def call_vision(
        self,
        screenshot: bytes,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Call GPT-4o vision API with screenshot and prompt.

        Args:
            screenshot: Screenshot image as bytes
            prompt: Text prompt describing the task
            temperature: Sampling temperature (default: 0.1 for deterministic)
            max_tokens: Maximum tokens in response (default: 500)

        Returns:
            Parsed JSON response from the model

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            APIError: If API call fails after retries
        """
        # Encode image
        base64_image = self.encode_image(screenshot)

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Call API with retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Calling vision API (attempt {attempt + 1}/{self.max_retries})")
                self.logger.debug(f"Prompt: {prompt[:200]}...")  # Log first 200 chars

                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Extract response content
                content = response.choices[0].message.content
                self.logger.debug(f"API Response: {content[:200]}...")  # Log first 200 chars

                # Parse JSON response
                try:
                    # Clean up markdown code blocks if present
                    cleaned_content = content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]  # Remove ```json
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content[3:]  # Remove ```
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                    cleaned_content = cleaned_content.strip()

                    result = json.loads(cleaned_content)
                    self.logger.info("Vision API call successful")
                    return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {e}")
                    self.logger.error(f"Raw response: {content}")
                    # If it's not JSON, return it wrapped in a dict
                    return {"raw_response": content, "error": "Invalid JSON"}

            except RateLimitError as e:
                self.logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = 60  # Wait 60 seconds
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached for rate limit")
                    raise

            except APITimeoutError as e:
                self.logger.warning(f"API timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 10 * (attempt + 1)  # Exponential backoff
                    self.logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached for timeout")
                    raise

            except APIError as e:
                error_message = str(e)
                self.logger.error(f"API error (attempt {attempt + 1}/{self.max_retries}): {error_message}")

                # Retry on specific errors
                if "timeout" in error_message.lower() or "connection" in error_message.lower():
                    if attempt < self.max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        self.logger.info(f"Retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error("Max retries reached for API error")
                        raise
                else:
                    # Don't retry for other errors
                    self.logger.error("Non-retryable API error")
                    raise

            except Exception as e:
                self.logger.error(f"Unexpected error calling vision API: {e}")
                raise

        # Should not reach here, but just in case
        raise APIError("Failed to get response from vision API after all retries")

    def call_vision_with_context(
        self,
        screenshot: bytes,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call vision API with context-aware prompt for test step execution.

        Builds a structured prompt with context information and requests
        JSON response with coordinates and action details.

        Args:
            screenshot: Screenshot image as bytes
            task: Current step text (e.g., "click on teststep named default_Measurement01")
            context: Optional context dict with keys like:
                - current_page: Current page description
                - previous_action: Last action performed
                - module: Current module/component
                - attempt: Retry attempt number

        Returns:
            Parsed JSON response with:
                - element_description: Description of found element
                - coordinates: {x, y} pixel coordinates
                - action_type: "click", "type", "dropdown"
                - value: Text value for type actions
                - confidence: Float 0-1 indicating confidence

        Example:
            >>> result = client.call_vision_with_context(
            ...     screenshot,
            ...     "click on teststep named default_Measurement01",
            ...     {"module": "Teststep", "previous_action": "navigated to page"}
            ... )
            >>> print(result['coordinates'])  # {'x': 450, 'y': 320}
        """
        context = context or {}

        # Build context-aware prompt
        prompt_parts = []

        # Add context information
        if context.get('module'):
            prompt_parts.append(f"Module: {context['module']}")
        if context.get('current_page'):
            prompt_parts.append(f"Current page: {context['current_page']}")
        if context.get('previous_action'):
            prompt_parts.append(f"Previous action: {context['previous_action']}")

        # Add task
        prompt_parts.append(f"\nTask: {task}")

        # Add instructions
        attempt = context.get('attempt', 1)
        if attempt == 1:
            detail_level = "standard"
        elif attempt == 2:
            detail_level = "enhanced - look more carefully at the relevant region"
        else:
            detail_level = "maximum - be very precise and specific"

        prompt_parts.append(f"\nDetail level: {detail_level}")

        prompt_parts.append("""
Find the UI element for this task and return a JSON object with:
{
    "element_description": "detailed description of the element found",
    "coordinates": {"x": <number>, "y": <number>},
    "action_type": "click" | "type" | "dropdown",
    "value": "text value if action_type is type, otherwise empty string",
    "confidence": <float between 0 and 1>
}

Guidelines:
- Choose the most prominent/obvious element if multiple matches exist
- Coordinates must be within viewport bounds (0-1920 width, 0-1080 height)
- Confidence should be >0.85 for reliable execution
- For type actions, include the text to be typed in the value field
- Return ONLY the JSON object, no additional text
""")

        prompt = "\n".join(prompt_parts)

        # Call vision API
        return self.call_vision(screenshot, prompt)

    def identify_login_selectors(
        self,
        screenshot: bytes
    ) -> Dict[str, Any]:
        """
        Use CV to identify the BEST CSS selectors for login page elements.

        This method analyzes a login page screenshot and returns SPECIFIC selectors
        that uniquely identify username field, password field, and login button.

        Args:
            screenshot: Screenshot image bytes of login page

        Returns:
            Dict with specific selectors:
            {
                "username_selector": "input[placeholder='Username']",
                "password_selector": "input[type='password']",
                "button_selector": "button:has-text('Login')",
                "confidence": 0.95
            }

        Example:
            >>> selectors = client.identify_login_selectors(screenshot)
            >>> page.locator(selectors['username_selector']).fill('mechanic')
        """
        prompt = """
You are analyzing a login page screenshot to identify the BEST CSS selectors for Playwright automation.

Your task: Find the MOST SPECIFIC and RELIABLE selectors for:
1. Username/Email input field
2. Password input field
3. Login/Submit button

GUIDELINES:
- Prefer specific selectors over generic ones
- Use attributes like placeholder, name, id, class if visible
- Use text content for buttons: button:has-text('Login')
- Avoid generic selectors like input[type='text'] if there are multiple text inputs
- Consider visual position (is it in a login form/card?)

Return ONLY this JSON format (no markdown, no code blocks):
{
    "username_selector": "<most specific CSS selector for username field>",
    "password_selector": "<most specific CSS selector for password field>",
    "button_selector": "<most specific CSS selector for login button>",
    "username_fallback": "<alternative selector if primary fails>",
    "password_fallback": "<alternative selector if primary fails>",
    "button_fallback": "<alternative selector if primary fails>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "Brief explanation of why these selectors were chosen"
}

EXAMPLES OF GOOD SELECTORS:
- input[placeholder='Username']
- input[name='username']
- input#username-field
- button.login-button
- button:has-text('Login')
- button[type='submit']

EXAMPLES OF BAD SELECTORS (too generic):
- input (matches ALL inputs)
- button (matches ALL buttons)
- input[type='text'] (if multiple text inputs exist)

Return the JSON now:
"""

        return self.call_vision(screenshot, prompt)

    def identify_step_selector(
        self,
        screenshot: bytes,
        step_text: str,
        custom_selector: Optional[str] = None,
        module: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use CV to identify the BEST selector for a specific test step.

        This method helps with:
        1. Choosing the right selector when multiple custom selectors exist
        2. Discovering selectors when no custom selector is available
        3. Providing context scoping for dynamic content

        Args:
            screenshot: Screenshot image bytes
            step_text: Test step description (e.g., "click on edit button of part default_testobject_01")
            custom_selector: Optional custom selector from JSON (if found)
            module: Optional module context

        Returns:
            Dict with selector strategy:
            {
                "selector": "tr:has-text('default_testobject_01') [data-editbtn]",
                "action": "click",
                "requires_scoping": true,
                "scope_selector": "tr:has-text('default_testobject_01')",
                "element_selector": "[data-editbtn]",
                "fallback_selectors": ["button:has-text('Edit')"],
                "confidence": 0.9,
                "reasoning": "..."
            }
        """
        prompt_parts = []

        # Add context
        prompt_parts.append("You are analyzing a web page to determine the BEST selector strategy for test automation.\n")

        if module:
            prompt_parts.append(f"Module context: {module}")

        prompt_parts.append(f"Task: {step_text}\n")

        # Add custom selector info if available
        if custom_selector:
            prompt_parts.append(f"Available custom selector from JSON: {custom_selector}\n")
            prompt_parts.append("Please verify if this selector is correct, or suggest improvements.\n")
        else:
            prompt_parts.append("NO custom selector available. You need to discover the best selector from the screenshot.\n")

        # Add instructions
        prompt_parts.append("""
Analyze the screenshot and provide selector strategy.

Consider:
1. **Scoping**: Does this action target a specific item in a list/table? (e.g., "default_testobject_01")
   - If yes, provide row/context scoping strategy

2. **Specificity**: Is there one element or multiple similar elements?
   - If multiple, how to uniquely identify the correct one?

3. **Selector types** (in order of preference):
   - Custom data attributes (e.g., [data-editbtn="EditBtn"])
   - Text-based (e.g., button:has-text('Save'))
   - ARIA attributes (e.g., [aria-label='Close'])
   - CSS classes (e.g., .save-button)
   - Standard HTML (e.g., button[type='submit'])

Return ONLY this JSON format (no markdown):
{
    "selector": "<complete Playwright selector>",
    "action": "click" | "fill" | "select",
    "value": "<text to type, if action is fill>",
    "requires_scoping": true | false,
    "scope_selector": "<row/context selector if scoping needed>",
    "element_selector": "<element selector within scope>",
    "fallback_selectors": ["<alternative 1>", "<alternative 2>"],
    "confidence": <float 0.0-1.0>,
    "reasoning": "Brief explanation of the strategy",
    "verification": "How to verify action succeeded (optional)"
}

Examples:

Task: "click on save button"
{
    "selector": "[data-savebtn='saveBtn']",
    "action": "click",
    "requires_scoping": false,
    "fallback_selectors": ["button:has-text('Save')", "button.save-btn"],
    "confidence": 0.95,
    "reasoning": "Found save button with custom data attribute"
}

Task: "click on edit button of part default_testobject_01"
{
    "selector": "tr:has-text('default_testobject_01') [data-editbtn]",
    "action": "click",
    "requires_scoping": true,
    "scope_selector": "tr:has-text('default_testobject_01')",
    "element_selector": "[data-editbtn]",
    "fallback_selectors": ["tr:has-text('default_testobject_01') button:has-text('Edit')"],
    "confidence": 0.92,
    "reasoning": "Edit button is in a table row. Need to scope to specific row containing 'default_testobject_01'"
}

Return the JSON now:
""")

        prompt = "\n".join(prompt_parts)
        return self.call_vision(screenshot, prompt)
