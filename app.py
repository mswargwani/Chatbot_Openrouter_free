import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (API keys, project names)
load_dotenv()

# Enable LangSmith tracing 
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Build the Streamlit UI
st.title("Langchain Demo With OpenRouter")
input_text = st.text_input("What question you have in mind? About IPL Ask Question from https://en.wikipedia.org/wiki/Indian_Premier_League till 2024 IPL")

# Define the Prompt emplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}")
    ]
)

# Initialize ChatOpenAI specifically for OpenRouter
llm = ChatOpenAI(
    base_url=f"{os.environ.get('OPENROUTER_BASE_URL')}/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    # It will pull your Llama 3 model or "openrouter/free" from your .env
    model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"), 
)
output_parser = StrOutputParser()

# Create the LCEL chain
chain = prompt | llm | output_parser

# 6. Execute the chain when the user provides input
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": input_text})
        st.write(response)