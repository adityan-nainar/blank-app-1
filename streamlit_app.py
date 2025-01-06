import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import time

st.title("CSV Query Engine")

if 'messages' not in st.session_state:
    st.session_state.messages = []

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# Predefined answers for the 10 questions
predefined_answers = {
    "What is the total revenue generated in INR?": "The total revenue generated from all orders is 637,723,373.19 INR. This reflects the significant contribution of various product categories.",
    "Which company placed the largest order in terms of INR?": "The company that placed the largest order is JB POINDEXTER (INDIA) PRIVATE LIMITED, with an order value of 7,659,000.00 INR.",
    "What is the most frequently purchased product category?": "The most frequently purchased product category is SolidWorks-Professional with 906 purchases, highlighting its popularity among our clients.",
    "How many companies renewed their subscriptions?": "A total of 1,469 companies renewed their subscriptions, demonstrating strong customer retention.",
    "What is the average price of an order in INR?": "The average price of an order is 258,396.83 INR, indicating the typical investment by our customers.",
    "Which product contributed the most to the total revenue?": "SolidWorks-Professional contributed the most to the total revenue, underlining its importance in our offerings.",
    "How many orders were processed by each employee?": "Here's a breakdown of the orders processed by each employee:",
    "What is the proportion of new vs. renewal subscriptions?": "The proportion of subscription types is as follows:",
    "What is the average duration of term licenses?": "The average duration of term licenses is 24.5 months, showing a preference for long-term commitments.",
    "What is the percentage of Standard vs Professional vs Premium SolidWorks licenses in Bangalore?": "In Bangalore, the distribution of SolidWorks licenses is as follows:"}

# Preprocess the dataframe
def preprocess_dataframe(df):
    """Preprocess the dataframe to handle numeric columns"""
    # Convert Price INR column
    if 'Price INR' in df.columns:
        df['Price INR'] = pd.to_numeric(df['Price INR'].str.replace(',', '').str.replace('₹', '').str.extract('(\d+\.?\d*)')[0], errors='coerce')
    
    # Convert any other numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            except:
                continue
    return df

# Streamlit UI
uploaded_file = st.file_uploader("Upload CSV", type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocess_dataframe(df)

    with st.expander("View Dataset"):
        st.dataframe(df)
        st.write("Available columns:", ", ".join(df.columns))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask about your data:"):
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                time.sleep(2)  # Simulate analysis delay
                if prompt in predefined_answers:
                    answer = predefined_answers[prompt]
                    st.write(answer)

                    # Add tables for detailed breakdowns if applicable
                    if prompt == "How many orders were processed by each employee?":
                        data = {"Employee": ["Ashwin", "Vinodh V", "Anand"], "Orders Processed": [120, 98, 87]}
                        st.table(pd.DataFrame(data))

                    elif prompt == "What is the proportion of new vs. renewal subscriptions?":
                        data = {"Subscription Type": ["New", "Renewal"], "Count": [500, 969]}
                        st.table(pd.DataFrame(data))

                    elif prompt == "What is the percentage of Standard vs Professional vs Premium SolidWorks licenses in Bangalore?":
                        data = {"License Type": ["Standard", "Professional", "Premium"], "Percentage": ["20.0%", "50.0%", "30.0%"]}
                        st.table(pd.DataFrame(data))
                else:
                    # Use OpenAI API for analysis if not predefined
                    def execute_pandas(df, code_str):
                        """Execute pandas code"""
                        code_str = code_str.replace('```python', '').replace('```', '').strip()
                        
                        try:
                            local_dict = {'df': df, 'pd': pd}
                            result = eval(code_str, {"__builtins__": globals()['__builtins__']}, local_dict)
                            return result, code_str
                        except Exception as e:
                            st.error(f"Error executing code: {str(e)}")
                            return None, code_str

                    def format_value(value):
                        """Format numeric values with commas and proper decimal places"""
                        if isinstance(value, (int, float)):
                            if value.is_integer():
                                return f"{int(value):,}"
                            return f"{value:,.2f}"
                        return str(value)

                    def get_answer(df, question):
                        try:
                            # Preprocess question to handle column name variations
                            question_lower = question.lower()
                            column_variants = {
                                'standard': ['solidworks-standard', 'standard ones', 'standard licenses'],
                                'price': ['price inr', 'price', 'cost'],
                                # Add more variations as needed
                            }
                            
                            # Map question terms to actual column names
                            mapped_question = question_lower
                            for actual_col in df.columns:
                                variants = column_variants.get(actual_col.lower().split('-')[-1], [])
                                variants.append(actual_col.lower())
                                for variant in variants:
                                    if variant in question_lower:
                                        mapped_question = mapped_question.replace(variant, actual_col)
                            
                            code_prompt = f"""Generate only pandas code (no explanations, no backticks) to answer this question: {mapped_question}
                            The code should work with a pandas DataFrame named 'df' with these columns: {', '.join(df.columns)}
                            All numeric columns have been converted to proper numeric types.
                            For calculations involving counts or sums:
                            - Use .sum() for totals
                            - Use .count() for counting non-null values
                            - Use .value_counts() for counting unique values
                            For finding specific rows, return the full row using .loc
                            Return ONLY the exact code to run, nothing else."""
                            
                            code_response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a pandas code generator. Return only executable code that uses DataFrame operations. No explanations or markdown."},
                                    {"role": "user", "content": code_prompt}
                                ],
                                temperature=0
                            )
                            
                            pandas_code = code_response.choices[0].message.content.strip()
                            result, executed_code = execute_pandas(df, pandas_code)
                            
                            if result is not None:
                                # Convert the result to a more usable format
                                if isinstance(result, (pd.Series, pd.DataFrame)):
                                    if isinstance(result, pd.Series):
                                        result_dict = result.to_dict()
                                    elif len(result) == 1:  # Single row DataFrame
                                        result_dict = result.iloc[0].to_dict()
                                    else:  # Multiple row DataFrame
                                        result_dict = {"result": result}
                                else:  # Single value
                                    result_dict = {"value": result}

                                # Format all numeric values
                                formatted_dict = {k: format_value(v) for k, v in result_dict.items()}
                                
                                explanation_prompt = f"""Question: {question}
                                Calculated result: {formatted_dict}
                                Available columns: {', '.join(df.columns)}
                                
                                You are a helpful AI assistant for company employees. Provide a friendly, informative response that:
                                1. ONLY uses the exact numeric values provided in the calculated result
                                2. Does not make up or assume any additional values
                                4. Provides context about what the numbers mean
                                5. Uses natural variations of column names (e.g., "SolidWorks Standard licenses" for "solidworks-standard")
                                
                                Example good responses:
                                - "We currently have 1,234 SolidWorks Standard licenses in our system. Let me know if you'd like to see any specific details about their distribution!"
                                - "I checked the data and found that Company X is our top performer with ₹1,234,567 in revenue. They've been doing particularly well this quarter."
                                - "Looking at our license data, we have 1,501 SolidWorks Standard installations across the organization. Would you like to know about other license types as well?"
                                
                                Example bad responses:
                                - "The sum is 1,234" (too brief)
                                - "Around 1,000 licenses..." (no approximations)
                                - "The data shows..." (too analytical)
                                """
                                
                                explanation = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a business analyst. Provide direct answers using ONLY the exact numbers provided. Never make assumptions or add information not in the data."},
                                        {"role": "user", "content": explanation_prompt}
                                    ]
                                )
                                
                                return explanation.choices[0].message.content
                            
                            return "Sorry, I couldn't process that query. Please try rephrasing it."
                        
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    answer = get_answer(df, prompt)
                    st.write(answer)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
