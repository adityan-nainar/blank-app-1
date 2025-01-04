import streamlit as st
import pandas as pd
import openai
from openai import OpenAI

st.title("CSV Query Engine")

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

def get_answer(df, question):
    data_info = f"Columns: {', '.join(df.columns)}\nSample data:\n{df.head().to_string()}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You're analyzing this dataset:\n{data_info}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

uploaded_file = st.file_uploader("Upload CSV", type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    
    question = st.text_input("Ask about your data:")
    if question and st.button("Get Answer"):
        answer = get_answer(df, question)
        st.write(answer)