# triage_services.py
#
# Medical Symptom Advisor - Triage Services Module
#
# This script implements core backend services that support dynamic triage guidance.
#
# Main Functionalities:
# - Retrieve symptom-specific guidelines from a structured JSON database.
# - Generate the next clarifying question based on previous user answers (Chain-of-Thought).
# - Use OpenAI's GPT model to generate structured final triage advice and safe care instructions.
#
# Core Components:
# - OpenAI client initialization using environment variables for API key and organization ID.
# - Symptom database lookup for structured triage rules.
# - Conversational logic for clarification question sequencing.
# - Prompt-based structured completion for final advice generation.
#
# Libraries Used:
# - openai
# - dotenv
# - json
# - Custom module: symptom_matcher.py (for semantic matching and AI response utilities)
#
# Author: Satish Swarnkar
# University of Texas at Austin


import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from symptom_matcher import match_symptom_semantic, get_ai_response, symptom_guidelines

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=os.getenv("OPENAI_ORG_ID")
)

GPT_MODEL = "gpt-4o-mini"  # You can change this to your preferred model

SYMPTOM_DB= "data/combined_symptom_database.json"

# Load symptom guidelines from a JSON file (ensure the file is in the same directory)
with open(SYMPTOM_DB, 'r') as file:
    symptom_guidelines = json.load(file)

def get_symptom_guidelines(symptom):
    """Get symptom guidelines from the database"""
    # Reuse your existing function to load the symptom database
    if symptom in symptom_guidelines:
        return symptom_guidelines[symptom]
    return None

def generate_follow_up_questions(symptom_data, previous_answers):
    """Generate the next question based on previous answers"""
    clarifying_questions = symptom_data.get('clarifying_questions', [])
    
    # If we've answered fewer questions than available, return the next one
    if len(previous_answers) < len(clarifying_questions):
        return clarifying_questions[len(previous_answers)]
    
    # If all questions have been answered, return None
    return None

def generate_gpt_response(symptom, answers):
    """Generate GPT-4 response with triage recommendations and advice"""
    symptom_data = symptom_guidelines.get(symptom, {})
    clarifying_questions = symptom_data.get('clarifying_questions', [])
    
    # Create a conversation that includes the symptom and all Q&A pairs
    conversation = f"Symptom: {symptom}\n\n"
    
    # Add the Q&A pairs
    for i, question in enumerate(clarifying_questions[:len(answers)]):
        conversation += f"Q: {question}\n"
        conversation += f"A: {answers[i]}\n\n"
    
    # Add the triage recommendations and conditions to flag for reference
    conversation += f"Conditions to Flag: {symptom_data.get('conditions_to_flag', [])}\n"
    conversation += f"Emergency Triage: {symptom_data.get('triage_recommendations', {}).get('emergency', '')}\n"
    conversation += f"Doctor Visit Triage: {symptom_data.get('triage_recommendations', {}).get('doctor_visit', '')}\n"
    conversation += f"Home Care Triage: {symptom_data.get('triage_recommendations', {}).get('monitor_at_home', '')}\n"
    conversation += f"Safe Advice: {symptom_data.get('safe_advice', '')}\n\n"
    
    # Add the prompt instruction
    conversation += "Based on the symptom and the patient's answers, provide:\n"
    conversation += "1. An assessment of the patient's condition\n"
    conversation += "2. The appropriate triage recommendation (emergency/doctor visit/home care)\n"
    conversation += "3. Safe advice for managing the symptoms\n"
    conversation += "Be compassionate but factual. If the answers indicate a condition to flag, highlight this clearly."
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing guidance based on symptom assessment."},
                {"role": "user", "content": conversation}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "Sorry, I couldn't generate advice at this time. Please try again later."
    

def process_symptom(user_input):
    """
    Process user symptom input and return matched symptom and data
    """
    # Use semantic search to match symptoms
    matched_symptom, symptom_data = match_symptom_semantic(user_input, OPENAI_API_KEY)
    return matched_symptom, symptom_data

def get_langchain_response(user_input):
    """
    Get an AI-generated response about the symptoms
    """
    return get_ai_response(user_input, OPENAI_API_KEY)

def get_all_symptoms():
    """
    Return a list of all symptoms in the database
    """
    return list(symptom_guidelines.keys())