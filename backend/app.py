import streamlit as st
import pandas as pd
import yaml
from openai import AzureOpenAI
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Defect Categorization",
    page_icon="🚗",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    with open("plcdtest_config.yaml", "r") as file:
        return yaml.safe_load(file)

# Load Excel data for few-shot examples
@st.cache_data
def load_training_data():
    """Load training data and return both limited and full dataset"""
    try:
        df = pd.read_excel("vehicle_inspections_with_defect_categories.xlsx")

        # Validate we have enough data
        if len(df) == 0:
            raise ValueError("Excel file is empty")

        # Get first 8 rows for training (or less if file has fewer rows)
        num_training_rows = min(8, len(df))
        training_df = df.head(num_training_rows)

        return training_df, df  # Return both: limited for training, full for examples
    except FileNotFoundError:
        raise FileNotFoundError("vehicle_inspections_with_defect_categories.xlsx not found")
    except Exception as e:
        raise Exception(f"Error loading Excel file: {str(e)}")

# Initialize Azure OpenAI client
@st.cache_resource
def get_openai_client():
    config = load_config()
    azure_config = config['azure_openai']

    client = AzureOpenAI(
        api_key=azure_config['api_key'],
        api_version=azure_config['api_version'],
        azure_endpoint=azure_config['endpoint']
    )
    return client, azure_config['deployment_gpt4o']

# Build few-shot prompt with examples
def build_few_shot_prompt(training_data):
    """Build the system prompt with few-shot examples from Excel data"""

    prompt = """You are an expert vehicle inspection analyst. Your task is to categorize vehicle defects into predefined categories, assign severity levels, and provide detailed reasoning using Chain of Thought.

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

    # Add few-shot examples from training data
    for idx, row in training_data.iterrows():
        example_num = idx + 1
        prompt += f"\n### Example {example_num}:\n"
        prompt += f"**Input Conditions:**\n"
        prompt += f"- Exterior/Body: {row['ExteriorBody']}\n"
        prompt += f"- Windshield: {row['Windshield']}\n"
        prompt += f"- Tire Condition: {row['TireCondition']}\n"
        prompt += f"- Fluid Levels: {row['FluidLevels']}\n"
        prompt += f"- Brake Condition: {row['BrakeCondition']}\n"
        prompt += f"- Warning Lights: {row['WarningLights']}\n"
        prompt += f"- AC System: {row['ACSystem']}\n"
        prompt += f"- Interior/Upholstery: {row['InteriorUpholstery']}\n\n"

        prompt += f"**Output:**\n"
        prompt += f"- Defect Codes: {row['DefectCodes']}\n"
        prompt += f"- Categories: {row['DefectCategories']}\n"
        prompt += f"- Severities: {row['DefectSeverities']}\n"
        prompt += f"- Defect Count: {row['DefectCount']}\n\n"

    prompt += """
## Your Task:
When given vehicle inspection conditions (can be 1-8 conditions), analyze each one using Chain of Thought reasoning:

### IMPORTANT: Only categorize actual defects!
- **Defects** are conditions like: "Major Damage", "Small Chip", "Needs Replacement", "Low", "Fair", "Check Engine Light", "Not Working", "Weak Airflow", "Tears", "Stains", etc.
- **NOT defects** are normal/good states like: "Excellent", "Good", "Clear", "OK", "Functional", "Clean", "New", or similar positive states, or "NaN"

### Analysis Steps:
1. Identify if there is an actual defect (ignore good/normal conditions)
2. Determine which category the defect belongs to
3. Assess the severity based on safety impact and urgency
4. Generate appropriate defect codes
5. Provide clear reasoning for your categorization

## Input Format:
The user may provide defects in various formats:
- **List format**: "ExteriorBody: Major Damage; ACSystem: Not Working"
- **Line-by-line format**: Each condition on a new line
- **Mixed format**: Any combination of the above
- **Partial input**: Only 1-7 conditions (not always 8)

Parse flexibly and focus ONLY on actual defects mentioned.

## Output Format:
For each defect found, provide:
- Defect Code (format: [CATEGORY]-[SEVERITY]-[NUMBER], e.g., EXT-CRT-001)
- Category (full name)
- Severity level
- Detailed reasoning explaining why this categorization was chosen

Please be thorough and use Chain of Thought reasoning to explain your decision-making process.
"""

    return prompt

# Parse flexible input format
def parse_input(input_text):
    """Parse flexible input - handles various formats"""
    # Try to detect if it's semicolon-separated format like "ExteriorBody: Major Damage; ACSystem: Not Working"
    if ';' in input_text:
        # Parse semicolon-separated format
        items = [item.strip() for item in input_text.split(';') if item.strip()]
        return items
    else:
        # Parse line-by-line format
        lines = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
        return lines

# Categorize defects using Azure OpenAI
def categorize_defects(input_items, system_prompt, client, deployment):
    """Send conditions to Azure OpenAI for categorization"""

    # Build user prompt with flexible input
    user_prompt = f"""Please analyze the following vehicle inspection conditions and categorize any defects:

**Inspection Input:**
{chr(10).join([f'{i+1}. {item}' for i, item in enumerate(input_items)])}

**Instructions:**
- Parse the input flexibly (it may contain field names like "ExteriorBody:" or just the condition)
- ONLY categorize actual defects (ignore good/normal conditions)
- Use Chain of Thought reasoning to analyze each defect
- Provide defect codes, categories, severities, and detailed reasoning

Remember: Focus ONLY on actual defects mentioned. Ignore any "Good", "Excellent", "OK", "Clean", etc.
"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Main Streamlit App
def main():
    st.title("🚗 Vehicle Defect Categorization System")
    st.markdown("**Powered by Azure OpenAI with Chain of Thought Reasoning**")

    st.markdown("---")

    # Load resources
    try:
        config = load_config()
        training_data, full_data = load_training_data()
        client, deployment = get_openai_client()
        system_prompt = build_few_shot_prompt(training_data)

        # Sidebar information
        with st.sidebar:
            st.header("📋 Information")
            st.markdown("""
            ### Defect Categories:
            - **EXT**: Exterior/Body
            - **WIN**: Windshield/Glass
            - **TIR**: Tires/Wheels
            - **FLD**: Fluids/Maintenance
            - **BRK**: Brakes/Safety
            - **ELC**: Electrical/Warning
            - **HVA**: HVAC/Climate
            - **INT**: Interior/Upholstery

            ### Severity Levels:
            - **Critical**: Immediate safety hazard
            - **Major**: Significant issue
            - **Moderate**: Should be addressed soon
            - **Minor**: Small/cosmetic issues
            """)

            st.markdown("---")
            st.info(f"✅ Using {len(training_data)} few-shot examples")
            st.success(f"✅ Connected to Azure OpenAI")

        # Input section
        st.header("📝 Enter Vehicle Defects")
        st.markdown("Enter defects in any format below. You can paste 1-8 conditions:")

        # Text area for input
        input_text = st.text_area(
            "Defect Conditions (Flexible Format)",
            height=200,
            placeholder="""Enter defects in any format:

Format 1 (List with labels):
ExteriorBody: Major Damage; ACSystem: Not Working; InteriorUpholstery: Tears

Format 2 (Line-by-line):
Major Damage
Small Chip
Needs Replacement
Check Engine Light

Format 3 (Mixed):
ExteriorBody: Major Damage
Windshield: Small Chip
Low
Check Engine Light

Note: Only actual defects will be categorized. "Good", "OK", "Clean" etc. will be ignored."""
        )

        # Categorize button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            categorize_btn = st.button("🔍 Categorize Defects", type="primary", use_container_width=True)

        # Process categorization
        if categorize_btn:
            if not input_text.strip():
                st.error("❌ Please enter defect conditions!")
            else:
                # Parse input flexibly
                input_items = parse_input(input_text)

                if len(input_items) == 0:
                    st.error("❌ No valid conditions found. Please enter at least one defect condition.")
                elif len(input_items) > 8:
                    st.warning(f"⚠️ You entered {len(input_items)} items. We'll analyze the first 8.")
                    input_items = input_items[:8]

                    with st.spinner("🔄 Analyzing defects using Chain of Thought reasoning..."):
                        result = categorize_defects(input_items, system_prompt, client, deployment)

                    # Display results
                    st.markdown("---")
                    st.header("📊 Categorization Results")

                    # Display input conditions
                    with st.expander("📋 Input Provided", expanded=False):
                        for i, item in enumerate(input_items, 1):
                            st.write(f"**{i}.** {item}")

                    # Display AI analysis
                    st.markdown("### 🤖 AI Analysis & Categorization")
                    st.markdown(result)

                    # Download button for results
                    st.download_button(
                        label="💾 Download Results",
                        data=f"Vehicle Defect Categorization Results\n\n{'='*50}\n\nInput Provided:\n{chr(10).join([f'{i+1}. {item}' for i, item in enumerate(input_items)])}\n\n{'='*50}\n\nAnalysis Results:\n{result}",
                        file_name="defect_categorization_results.txt",
                        mime="text/plain"
                    )
                else:
                    with st.spinner("🔄 Analyzing defects using Chain of Thought reasoning..."):
                        result = categorize_defects(input_items, system_prompt, client, deployment)

                    # Display results
                    st.markdown("---")
                    st.header("📊 Categorization Results")

                    # Display input conditions
                    with st.expander("📋 Input Provided", expanded=False):
                        for i, item in enumerate(input_items, 1):
                            st.write(f"**{i}.** {item}")

                    # Display AI analysis
                    st.markdown("### 🤖 AI Analysis & Categorization")
                    st.markdown(result)

                    # Download button for results
                    st.download_button(
                        label="💾 Download Results",
                        data=f"Vehicle Defect Categorization Results\n\n{'='*50}\n\nInput Provided:\n{chr(10).join([f'{i+1}. {item}' for i, item in enumerate(input_items)])}\n\n{'='*50}\n\nAnalysis Results:\n{result}",
                        file_name="defect_categorization_results.txt",
                        mime="text/plain"
                    )

        # Show example
        with st.expander("💡 View Example Inputs", expanded=False):
            try:
                # Show a realistic example from the actual defect list
                st.markdown("**Example 1: List format (from training data)**")
                if len(training_data) > 0:
                    example_row = training_data.iloc[0]
                    st.code(example_row['DefectsList'])
                else:
                    st.warning("No training data available for example 1")

                st.markdown("**Example 2: Line-by-line format**")
                st.code("""Major Damage
Small Chip
Needs Replacement
Check Engine Light
Not Working""")

                st.markdown("**Example 3: Fewer defects (from full dataset)**")
                # Search in full dataset for a record with 2-3 defects
                few_defects = full_data[full_data['DefectCount'].isin([2, 3])]
                if len(few_defects) > 0:
                    example_few = few_defects.iloc[0]
                    st.code(example_few['DefectsList'])
                    st.caption(f"(This example has {example_few['DefectCount']} defects)")
                else:
                    # Fallback: show any record with fewer than 6 defects
                    fewer_defects = full_data[full_data['DefectCount'] < 6]
                    if len(fewer_defects) > 0:
                        example_few = fewer_defects.iloc[0]
                        st.code(example_few['DefectsList'])
                        st.caption(f"(This example has {example_few['DefectCount']} defects)")
                    else:
                        st.info("No examples with fewer defects available")
            except Exception as ex:
                st.warning(f"Could not load examples: {str(ex)}")

    except FileNotFoundError as e:
        st.error(f"❌ Configuration file or Excel file not found: {str(e)}")
        st.info("Please ensure `plcdtest_config.yaml` and `vehicle_inspections_with_defect_categories.xlsx` are in the same directory as this script.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
