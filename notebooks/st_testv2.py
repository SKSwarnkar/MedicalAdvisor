from dotenv import load_dotenv
import os

import openai
from openai import OpenAI
import re

import json
import streamlit as st

from ph0_2 import get_medical_advice
import streamlit as st

# Streamlit UI
st.title("Medical Symptom Advisor")

symptom_input = st.text_input("Enter a symptom:")

if symptom_input:
    # Use the function from the notebook
    advice = get_medical_advice(symptom_input)
    st.write(advice)