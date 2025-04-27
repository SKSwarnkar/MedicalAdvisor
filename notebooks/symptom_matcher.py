# symptom_matcher.py
#
# Medical Symptom Advisor - Symptom Matching and LangChain Initialization
#
# This script powers the semantic symptom matching system for the Medical Symptom Advisor.
#
# Main Functionalities:
# - Load a structured symptom database from JSON.
# - Create vector embeddings for symptoms using OpenAI models.
# - Build a FAISS vectorstore for efficient similarity search.
# - Initialize a LangChain Conversational Retrieval Chain to manage symptom clarification dialogues.
#
# Core Components:
# - OpenAIEmbeddings: Generates semantic vectors from symptom guidelines.
# - FAISS Vectorstore: Stores embeddings for fast nearest-neighbor search.
# - ConversationalRetrievalChain: Powers dynamic multi-turn conversations based on retrieved documents.
# - ConversationBufferMemory: Maintains dialogue state between user and system.
# - PromptTemplate: Structures how questions and clarifications are asked.
#
# Libraries Used:
# - openai
# - dotenv
# - langchain_openai
# - langchain_community.vectorstores (FAISS)
# - langchain.chains (ConversationalRetrievalChain)
# - langchain.memory (ConversationBufferMemory)
#
# Author: Satish Swarnkar
# University of Texas at Austin


import json
import os
import openai
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-4o-mini"  # You can change this to your preferred model

# Load symptom database
SYMPTOM_DB = "data/combined_symptom_database.json"

# Load symptom guidelines from a JSON file
with open(SYMPTOM_DB, 'r') as file:
    symptom_guidelines = json.load(file)

# Initialize OpenAI embeddings
embeddings = None
vectorstore = None
conversation_chain = None

def initialize_langchain(api_key):
    """Initialize LangChain components with your OpenAI API key"""
    global embeddings, vectorstore, conversation_chain
    
    # Create OpenAI embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Convert symptom guidelines to documents for vector storage
    documents = []
    for symptom, details in symptom_guidelines.items():
        # Format the document with all relevant symptom information
        content = f"Symptom: {symptom}\n"
        content += f"Clarifying Questions: {', '.join(details['clarifying_questions'])}\n"
        content += f"Conditions to Flag: {', '.join(details['conditions_to_flag'])}\n"
        content += f"Emergency Triage: {details['triage_recommendations']['emergency']}\n"
        content += f"Doctor Visit Triage: {details['triage_recommendations']['doctor_visit']}\n"
        content += f"Home Care Triage: {details['triage_recommendations']['monitor_at_home']}\n"
        content += f"Safe Advice: {details['safe_advice']}"
        
        doc = Document(
            page_content=content,
            metadata={"symptom": symptom}
        )
        documents.append(doc)
    
    # Create vector store from documents
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create custom prompt template for medical advisor
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""You are a medical triage assistant. Use the symptom information below to help the patient:

{context}

Current conversation:
{chat_history}

Patient's latest message: {question}

IMPORTANT:
1. First, identify which symptom matches the patient's description
2. Ask EXACTLY ONE clarifying question, then WAIT for a response
3. After receiving an answer, ask a DIFFERENT question
4. After asking 3 TOTAL questions, provide a final conclusion with:
   - Any conditions to flag
   - Appropriate triage recommendation (emergency/doctor/home care)
   - Safe advice from the database

DO NOT repeat questions - check the conversation history to see what you've already asked.
NEVER ask multiple questions at once.
Always count how many questions you've asked and provide a conclusion after the third question.

Your response:"""
    )
    
    # Initialize ChatGPT model
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model=GPT_MODEL,
        temperature=0.2
    )
    
    # Create conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return True

def match_symptom_semantic(user_input, api_key=None):
    """
    Use semantic search to match user input to symptoms
    """
    global embeddings, vectorstore, conversation_chain
    
    # Initialize if not already done
    if embeddings is None or vectorstore is None or conversation_chain is None:
        if api_key is None:
            return None, "OpenAI API key required for semantic search"
        initialize_langchain(api_key)
    
    # Use the vectorstore to find similar symptoms
    docs = vectorstore.similarity_search(user_input, k=1)
    if docs:
        # Extract symptom name from metadata
        matched_symptom = docs[0].metadata.get("symptom")
        return matched_symptom, symptom_guidelines.get(matched_symptom)
    
    return None, None

def get_ai_response(user_input, api_key=None):
    """
    Get AI-generated response using LangChain and ChatGPT
    """
    global conversation_chain
    
    # Initialize if not already done
    if conversation_chain is None:
        if api_key is None:
            return "OpenAI API key required for AI response"
        initialize_langchain(api_key)
    
    # Get response from conversation chain
    try:
        response = conversation_chain.invoke({"question": user_input})
        return response["answer"]
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return f"I encountered an error processing your request. Please try again or describe your symptoms differently."

