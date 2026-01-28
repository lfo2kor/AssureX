import yaml
from openai import AzureOpenAI
import pandas as pd

# Load config
with open("plcdtest_config.yaml", "r") as file:
    config = yaml.safe_load(file)

azure_config = config['azure_openai']

# Initialize client
client = AzureOpenAI(
    api_key=azure_config['api_key'],
    api_version=azure_config['api_version'],
    azure_endpoint=azure_config['endpoint']
)

# Load training data
df = pd.read_excel("vehicle_inspections_with_defect_categories.xlsx")
training_data = df.head(8)

# Build system prompt (simplified version)
system_prompt = """You are an expert vehicle inspection analyst. Your task is to categorize vehicle defects into predefined categories, assign severity levels, and provide detailed reasoning using Chain of Thought.

## Defect Categories:
1. **Exterior/Body (EXT)** - Body damage, paint issues, exterior panels
2. **Windshield/Glass (WIN)** - Windshield chips, cracks, glass damage
3. **Tires/Wheels (TIR)** - Tire wear, wheel damage, pressure issues
4. **Fluids/Maintenance (FLD)** - Oil, coolant, brake fluid levels
5. **Brakes/Safety (BRK)** - Brake condition, brake pads, brake system
6. **Electrical/Warning (ELC)** - Warning lights, electrical issues
7. **HVAC/Climate (HVA)** - AC system, heating, climate control
8. **Interior/Upholstery (INT)** - Seats, interior condition, wear and tear

## Severity Levels:
- **Critical** - Immediate safety hazard, vehicle unsafe to operate
- **Major** - Significant issue requiring prompt attention
- **Moderate** - Should be addressed soon but not urgent
- **Minor** - Cosmetic or small issues

## Few-Shot Examples:
"""

# Add few examples
for idx in range(min(3, len(training_data))):
    row = training_data.iloc[idx]
    system_prompt += f"\n### Example {idx+1}:\n"
    system_prompt += f"**Input:** {row['DefectsList']}\n"
    system_prompt += f"**Output:**\n"
    system_prompt += f"- Defect Codes: {row['DefectCodes']}\n"
    system_prompt += f"- Categories: {row['DefectCategories']}\n"
    system_prompt += f"- Severities: {row['DefectSeverities']}\n\n"

system_prompt += """
## Your Task:
Analyze the defects using Chain of Thought reasoning and provide:
- Defect Code (format: [CATEGORY]-[SEVERITY]-[NUMBER])
- Category (full name)
- Severity level
- Detailed reasoning

Focus ONLY on actual defects mentioned.
"""

# Test input
test_input = "Windshield: Small Chip; BrakeCondition: Fair; WarningLights: Check Engine Light; InteriorUpholstery: Minor Wear"

user_prompt = f"""Please analyze the following vehicle inspection conditions and categorize any defects:

**Inspection Input:**
{test_input}

**Instructions:**
- Parse the input flexibly
- ONLY categorize actual defects (ignore good/normal conditions)
- Use Chain of Thought reasoning to analyze each defect
- Provide defect codes, categories, severities, and detailed reasoning
"""

print("=" * 80)
print("INPUT:")
print(test_input)
print("\n" + "=" * 80)
print("SENDING TO AZURE OPENAI...")
print("=" * 80)

# Call Azure OpenAI
response = client.chat.completions.create(
    model=azure_config['deployment_gpt4o'],
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3,
    max_tokens=2000
)

result = response.choices[0].message.content

print("\nOUTPUT:")
print("=" * 80)
print(result)
print("=" * 80)
