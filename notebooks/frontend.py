import streamlit as st
from backend import generate_guideline_prompt, get_gpt_response  # Import backend functions

# Streamlit UI setup
st.title("Medical Symptom Advisor")
st.write("Enter a symptom, and get recommendations for triage and clarifying questions.")

# Create a text input field for the symptom
symptom_input = st.text_input("Enter a symptom:")

# Function to display the result based on the input symptom
def display_result(symptom):
    prompt = generate_guideline_prompt(symptom)  # Generate prompt for symptom using backend function
    if "Sorry" in prompt:  # Check if symptom was found
        st.write(prompt)  # If not found, display a message
    else:
        gpt_response = get_gpt_response(prompt)  # Get response from backend
        st.write(gpt_response)  # Display the response

# Call the function to display the result when the user inputs a symptom
if symptom_input:
    display_result(symptom_input)

# Front End Code