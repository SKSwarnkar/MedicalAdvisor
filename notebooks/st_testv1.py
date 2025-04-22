import streamlit as st

# Title of the app
st.title("Streamlit Test App")

# Input text box
name = st.text_input("Enter your name:")

# Button to display greeting
if st.button("Greet"):
    st.write(f"Hello, {name}!")
    

# Display a simple text
st.write("Hello, Streamlit is working!")

# You can add more interactive widgets, such as sliders or buttons
number = st.slider("Pick a number", 0, 100)
st.write(f"Selected number: {number}")
