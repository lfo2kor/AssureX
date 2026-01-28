import os
import requests
import yaml
import json
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
from config import settings


load_dotenv()
router = APIRouter()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
# JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Load YAML configuration
# CONFIG_PATH = r"C:\Idea Projects\PLCD_TA_Team\PLCD_TA_Team\plcdtestassistant.yaml"
# CONFIG_PATH = r"C:\Idea Projects\AI_Test_Assist\plcdtestassistant.yaml"
CONFIG_PATH = r"C:\AssureX\backend\plcdtestassistant.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Initialize Azure OpenAI client
azure_config = config["azure_openai"]
client = AzureOpenAI(
    api_key=azure_config["api_key"],
    api_version=azure_config["api_version"],
    azure_endpoint=azure_config["endpoint"]
)

def parse_jira_ticket_with_llm(ticket_data: dict) -> dict:
    """
    Parse Jira ticket using LLM with few-shot prompting
    """
    jira_config = config["jira_agent"]
    
    # Build prompt with few-shot examples
    prompt = f"""
{jira_config['system_prompt']}

{jira_config['format_examples']}

NOW PARSE THIS TICKET:

Ticket ID: {ticket_data['ticket_id']}
Title: {ticket_data['title']}
Description:
{ticket_data['description']}

Return ONLY valid JSON, no markdown, no code blocks, no other text.
"""

    try:
        response = client.chat.completions.create(
            model=jira_config["model"],
            messages=[
                {"role": "system", "content": jira_config["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            temperature=jira_config["temperature"],
            max_tokens=jira_config["max_tokens"]
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        parsed = json.loads(content)
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"LLM Response: {content}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")
    except Exception as e:
        print(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")


@router.get("/jira/{ticket_id}")
def get_jira_ticket(ticket_id: str):
    """
    Fetch Jira ticket from Jira Server and parse with LLM
    """
    print(f"Received request for Jira ticket: {ticket_id}")  # Add this line
    # return {"ticket_id": ticket_id, "title": "Dummy Title"}
    url = f"{JIRA_BASE_URL}/rest/api/2/issue/{ticket_id}"
    headers = {
        "Authorization": f"Bearer {JIRA_API_TOKEN}",
        "Accept": "application/json"
    }
    
    # Fetch from Jira
    response = requests.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        print("Jira API error:", response.status_code, response.text)
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Failed to fetch Jira ticket: {response.text}"
        )
    
    # Extract fields
    jira_data = response.json()
    fields = jira_data.get("fields", {})
    
    raw_description = fields.get("description", "")
    summary = fields.get("summary", "")
    
    # Clean HTML from description if present
    import re
    clean_description = re.sub(r'<[^>]+>', '', raw_description) if raw_description else ""
    
    print(f"Fetched ticket: {ticket_id}")
    print(f"Summary: {summary}")
    print(f"Description length: {len(clean_description)}")
    
    # Parse with LLM using few-shot prompting
    parsed_ticket = parse_jira_ticket_with_llm({
        "ticket_id": ticket_id,
        "title": summary,
        "description": clean_description
    })
    
    # Return parsed ticket with original description for display
    return {
        **parsed_ticket,
        "raw_description": clean_description,  # For UI display
        "original_summary": summary
    }







