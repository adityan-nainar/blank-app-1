import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def load_data(file):
    """Loads the uploaded dataset."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def generate_code(question, context):
    """Generates Python code to answer the user's question using OpenAI."""
    try:
        prompt = f"""
Based on this dataset summary: {context}

Question: {question}

Generate ONLY executable Python code with NO explanations or text outside of code comments. The code should:
1. Use the pre-loaded 'dataset' DataFrame
2. Create a 'result' dictionary with exactly these keys:
   - 'main_result': The primary answer
   - 'context': A dictionary of relevant supporting information
   - 'explanation': A text explanation of the findings
3. NOT include any print statements
4. NOT include any text or explanations after the code
5. NOT include any markdown formatting

Example structure:
# Analysis code here
revenue = dataset.something()

# Create result dictionary
result = {{
    'main_result': value,
    'context': {{key: value}},
    'explanation': f"Text explanation"
}}

IMPORTANT: Return ONLY the executable code. No explanations or text after the code.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code generator that produces ONLY executable Python code. Never include explanations, markdown, or non-code text in your response."
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        # Clean the response
        code = response.choices[0].message.content.strip()
        
        # Remove any markdown formatting
        code = code.replace("```python", "").replace("```", "")
        
        # Remove any trailing explanations (anything after the last code line)
        code_lines = code.split('\n')
        cleaned_lines = []
        for line in code_lines:
            # Skip empty lines and lines that look like explanations
            if line.strip() and not line.startswith('This') and not line.startswith('The'):
                cleaned_lines.append(line)
        
        # Remove any trailing lines that might be explanations
        while cleaned_lines and not any(char in cleaned_lines[-1] for char in '=[]{}()+-*/'):
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)
    except Exception as e:
        st.error(f"Error querying OpenAI API: {str(e)}")
        return None

def execute_code(code, dataset):
    """Executes the generated Python code in a safe environment."""
    try:
        # Create a namespace with required objects
        namespace = {
            'pd': pd,
            'np': np,
            'dataset': dataset,
            'result': None
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Return the result
        if 'result' in namespace:
            return namespace['result']
        else:
            return {
                'main_result': "No result found",
                'context': {},
                'explanation': "The analysis did not produce any results."
            }
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")
        return None

def display_result(result):
    """Displays the analysis results in a structured format with wider tables."""
    if isinstance(result, dict) and 'main_result' in result:
        # Display the explanation first
        if 'explanation' in result:
            st.write("### 📊 Analysis")
            st.write(result['explanation'])
        
        # Display the main result
        st.write("### 🎯 Main Result")
        if isinstance(result['main_result'], pd.DataFrame):
            st.dataframe(result['main_result'], width=800)  # Set explicit width
        elif isinstance(result['main_result'], (pd.Series, dict)):
            # Convert to DataFrame for better display
            main_df = pd.DataFrame([result['main_result']]).T
            main_df.columns = ['Value']
            st.dataframe(main_df, width=800)  # Set explicit width
        else:
            st.write(result['main_result'])
        
        # Display additional context if available
        if 'context' in result and result['context']:
            st.write("### 📌 Additional Information")
            # Format numerical values in context
            formatted_context = {}
            for key, value in result['context'].items():
                if isinstance(value, (int, float)):
                    formatted_context[key] = f"{value:,.2f}"
                elif isinstance(value, dict):
                    # Format nested dictionary values
                    formatted_nested = {k: f"{v:,.2f}" if isinstance(v, (int, float)) else v 
                                     for k, v in value.items()}
                    formatted_context[key] = formatted_nested
                else:
                    formatted_context[key] = value
                    
            # If we have nested dictionaries, flatten them for display
            flat_context = {}
            for key, value in formatted_context.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_context[f"{key} - {sub_key}"] = sub_value
                else:
                    flat_context[key] = value
                    
            context_df = pd.DataFrame([flat_context]).T
            context_df.columns = ['Value']
            
            # Apply custom styling to make the table wider
            st.dataframe(
                context_df,
                width=800,  # Set explicit width
                height=None  # Auto-adjust height
            )
    else:
        st.write(result)

# Streamlit app layout
st.title("Data Analysis Chatbot")

# Add description
st.markdown("""
This chatbot helps you analyze your dataset by answering questions in natural language.
Simply upload your CSV file and ask questions about your data!
""")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type="csv")

if uploaded_file:
    dataset = load_data(uploaded_file)
    if dataset is not None:
        st.success("Dataset loaded successfully!")
        
        # Show dataset info
        st.write("### Dataset Preview")
        st.dataframe(dataset.head())
        
        # Show dataset statistics
        # st.write("### Dataset Information")
        # st.write(f"- Rows: {dataset.shape[0]}")
        # st.write(f"- Columns: {dataset.shape[1]}")
        # st.write("- Column names:", ", ".join(dataset.columns.tolist()))
        
        # Generate a detailed dataset summary
        summary = f"""
        Dataset has {dataset.shape[0]} rows and {dataset.shape[1]} columns.
        Columns: {', '.join(dataset.columns.tolist())}
        Data types: {', '.join([f'{col}: {str(dataset[col].dtype)}' for col in dataset.columns])}
        """
        
        # User question input
        user_question = st.text_input("Ask a question about your dataset:")
        
        if user_question:
            with st.spinner("Analyzing your question..."):
                code = generate_code(user_question, summary)
                
                if code:
                    st.write("### Generated Python Code")
                    st.code(code, language='python')
                    
                    with st.spinner("Executing analysis..."):
                        result = execute_code(code, dataset)
                        
                        if result is not None:
                            display_result(result)
                else:
                    st.error("Could not generate code. Please try rephrasing your question.")

# Instructions for users
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a CSV dataset using the file uploader
    2. Review the dataset preview and information
    3. Ask questions about your dataset in plain English
    4. View the generated code and detailed analysis results
    
    Example questions:
    - What is the average value of [column]?
    - Which [category] has the highest [metric]?
    - Show me the trend of [metric] over time
    - What is the distribution of [column]?
    - How does [metric1] correlate with [metric2]?
    """)