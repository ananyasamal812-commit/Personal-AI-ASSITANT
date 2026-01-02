import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# Tools
@tool
def summarize_text(text: str) -> str:
    """Summarize input text"""
    return llm.predict(f"Summarize this:\n{text}")

@tool
def generate_notes(topic: str) -> str:
    """Generate notes on a topic"""
    return llm.predict(f"Generate structured notes on {topic}")

tools = [summarize_text, generate_notes]

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# Vector memory
embeddings = OpenAIEmbeddings()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = FAISS.from_texts([], embeddings)

def store_memory(text):
    st.session_state.vectorstore.add_texts([text])

def retrieve_memory(query):
    docs = st.session_state.vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# Streamlit UI
st.title("🧠 Personal AI Assistant")

query = st.text_input("Ask me anything")

if st.button("Run"):
    if query:
        context = retrieve_memory(query)
        response = agent.run(query)
        store_memory(response)
        st.success("Response:")
        st.write(response)