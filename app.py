import streamlit as st
from langchain_groq import ChatGroq
import os
import pandas as pd
from pandasai import Agent
from dotenv import load_dotenv

groq_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model="llama3-8b-8192")

st.title("Data Analysis of your data")
uploaded_file=st.sidebar.file_uploader("Upload a CSV file",type=["CSV"])

if uploaded_file is not None:
    data=pd.read_csv( uploaded_file)
    st.write(data.head(3))

    agent = Agent(data,config={"llm":llm})
    prompt= st.text_input("Enter your prompt:")

    if st.button("Submit"):
        if prompt:
            with st.spinner("Generating response .."):
                st.write(agent.chat(prompt))
