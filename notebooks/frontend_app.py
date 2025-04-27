# frontend_app.py
#
# Medical Symptom Advisor - Frontend Application
#
# This script builds the Streamlit-based user interface for the Medical Symptom Advisor system.
#
# Main Functionalities:
# - Allow users to input symptoms in free text.
# - Perform semantic matching of symptoms using FAISS-based embedding search.
# - Engage users in dynamic multi-turn clarification dialogs using Chain-of-Thought reasoning.
# - Display triage recommendations, safe advice, and enrich results with real-time external medical data (OpenFDA, PubMed).
# - Provide interactive visualizations such as embedding plots, similarity heatmaps, and symptom networks.
#
# Core Components:
# - OpenAI API Key Management via .env files and Streamlit input.
# - Symptom matching powered by LangChain agents (symptom_matcher module).
# - Triage question-answering and safe advice generation (triage_services module).
# - Visualization support through display_charts and visualization modules.
#
# Libraries Used:
# - streamlit
# - openai
# - dotenv
# - Custom modules: symptom_matcher.py, triage_services.py, symptom_analytics_charts.py, visualization.py
#
# Author: Satish Swarnkar
# University of Texas at Austin



import streamlit as st

import os
import openai
from dotenv import load_dotenv

from symptom_matcher import initialize_langchain
        
from triage_services import get_symptom_guidelines, generate_follow_up_questions, generate_gpt_response

from triage_services import symptom_guidelines
from triage_services import process_symptom, get_langchain_response, get_all_symptoms

from symptom_analytics_charts import create_and_display_chart

from visualization import visualize_embeddings, create_cosine_similarity_heatmap, create_top_similar_symptoms_chart
from visualization import get_all_symptom_embeddings, create_symptom_network, analyze_embedding_dimensions, create_triage_severity_distribution

load_dotenv()

# Set OpenAI API key from environment variable if not already set
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
# Check for API key
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
# API Key input (only if not set in environment)
if not st.session_state.openai_api_key:
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        st.session_state.openai_api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully!")

# Streamlit UI setup
st.title("Medical Symptom Advisor")
st.write("Enter a symptom, and get AI-powered recommendations and guidance.")

# Set up tabs for different modes

tab0, tab1, tab2, tab_symptom_analytics, tab_nlp_analytics = st.tabs([
    "GPT Direct Mode", "Simple Mode", "AI Chat Mode", "Symptom Analytics",
    "NLP Analytics"
    ])


# Tab 0: GPT Direct Mode (Sequential Questions)
with tab0:
    st.header("Chain-Of-Thought Reasoning: GPT Direct Mode")
    
    # Add explanation
    st.info("""
    This mode takes you through a structured interview:
    1. Enter your symptom directly
    2. Answer each clarifying question one by one
    3. Receive a GPT-generated assessment and advice
    """)
    
    # Step 1: Get symptom input
    symptom_input = st.text_input("Enter a specific symptom:", key="direct_symptom")
    
    # Display all available symptoms
    with st.expander("Available symptoms in database"):
        st.write(", ".join(get_all_symptoms()))
    
    # Step 2: Initialize or reset session state
    if "answers" not in st.session_state:
        st.session_state.answers = []
    
    # Clear the answers if the symptom changes
    if symptom_input != st.session_state.get('last_symptom', ''):
        st.session_state.answers = []  # Reset answers when symptom changes
    
    st.session_state.last_symptom = symptom_input  # Store the last entered symptom
    
    # Reset button
    if st.button("Start Over"):
        st.session_state.answers = []
        # st.experimental_rerun()
        st.rerun()
    
    # Step 3: Process the symptom and ask questions
    if symptom_input:
        # Get symptom data
        symptom_data = get_symptom_guidelines(symptom_input)
        
        if symptom_data:
            st.success(f"Processing: {symptom_input}")
            
            # Display current progress
            progress = len(st.session_state.answers)
            total_questions = len(symptom_data["clarifying_questions"])
            
            # Show progress bar
            if total_questions > 0:
                st.progress(progress / total_questions)
            
            # If we haven't answered all questions yet
            if progress < total_questions:
                # Get the next question
                next_question = generate_follow_up_questions(symptom_data, st.session_state.answers)
                
                # Ask the question
                col1, col2 = st.columns([3, 1])
                with col1:
                    answer = st.text_input(
                        f"Question {progress+1}/{total_questions}: {next_question}", 
                        key=f"q{progress}"
                    )
                with col2:
                    submit = st.button("Submit Answer")
                
                # Process the answer
                if submit and answer:
                    st.session_state.answers.append(answer)
                    # st.experimental_rerun()
                    st.rerun()
            
            # If we've answered all questions, generate the response
            if len(st.session_state.answers) == total_questions:
                st.subheader("Assessment Complete!")
                
                # Show a summary of questions and answers
                with st.expander("Review Your Answers"):
                    for i, question in enumerate(symptom_data["clarifying_questions"]):
                        if i < len(st.session_state.answers):
                            st.write(f"**Q: {question}**")
                            st.write(f"A: {st.session_state.answers[i]}")
                
                # Generate and display the GPT response
                with st.spinner("Generating assessment..."):
                    gpt_response = generate_gpt_response(symptom_input, st.session_state.answers)
                
                st.markdown("### AI Assessment and Recommendations:")
                st.markdown(gpt_response)
                from advisor_enrichment import medical_advisor_enrichment
                medical_advisor_enrichment(gpt_response)
                
                # Add a note about the source of this information
                st.caption("This assessment is generated using GPT based on your answers and our medical database.")
        else:
            st.error(f"Sorry, '{symptom_input}' is not in our database. Please check the list of available symptoms.")

# Simple mode (original functionality)
with tab1:
    st.header("Semantic Symptom Matching & Recommendations")
    
    # Step 1: Get symptom input (free-text)
    symptom_input = st.text_input("Describe your symptom:")
    
    # Display all available symptoms
    with st.expander("Available symptoms in database"):
        st.write(", ".join(get_all_symptoms()))
    
    # Step 2: Display results after symptom is entered
    if symptom_input:
        matched_symptom, symptom_data = process_symptom(symptom_input)
    
        if matched_symptom and symptom_data:
            st.success(f"Matched Symptom: {matched_symptom}")
            
            st.subheader("Clarifying Questions:")
            for question in symptom_data["clarifying_questions"]:
                st.write(f"- {question}")
            
            st.subheader("Conditions to Flag:")
            for condition in symptom_data["conditions_to_flag"]:
                st.warning(condition)
            
            st.subheader("Triage Recommendations:")
            st.error(f"Emergency: {symptom_data['triage_recommendations']['emergency']}")
            st.warning(f"Doctor Visit: {symptom_data['triage_recommendations']['doctor_visit']}")
            st.info(f"Home Care: {symptom_data['triage_recommendations']['monitor_at_home']}")
            
            st.subheader("Safe Advice:")
            st.write(symptom_data["safe_advice"])
            from advisor_enrichment import medical_advisor_enrichment
            medical_advisor_enrichment(symptom_data["safe_advice"])
        else:
            st.error("Sorry, we couldn't match your symptom. Please try again or use the AI Chat mode.")

# AI Chat mode (LangChain + ChatGPT)
with tab2:
    st.header("Agentic AI-Powered Symptom Chat")
    
    # Add explanation
    st.info("""
    This mode uses AI to conduct a brief symptom assessment:
    1. Describe your main symptom or health concern
    2. The AI will ask you up to 3 clarifying questions
    3. After these questions, it will provide a triage recommendation and advice
    """)
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your medical symptom advisor. Please describe what you're experiencing, and I'll ask a few specific questions to provide appropriate advice."}
        ]
    
    # Add this to your frontend.py in the tab2 section (AI Chat Mode)
    if "question_counter" not in st.session_state:
        st.session_state.question_counter = 0

        
    # Add reset button to start a new conversation
    if st.button("Start New Conversation"):
        # Reset session messages
        st.session_state.question_counter = 0
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your medical symptom advisor. Please describe what you're experiencing, and I'll ask a few specific questions to provide appropriate advice."}
        ]
        
        # Reset LangChain memory
        if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
            try:
                initialize_langchain(st.session_state.openai_api_key)
                st.success("Chat history has been reset. You can start a new conversation.")
            except Exception as e:
                st.error(f"Error resetting conversation: {e}")
        
        # Force refresh
        st.rerun()
            
    # Create a container for the chat
    chat_container = st.container()
    
    # Handle user input
    if st.session_state.openai_api_key:
        user_input = st.chat_input("Describe your symptoms...")
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Check if user was answering a question
            last_ai_message = next((msg for msg in reversed(st.session_state.messages) 
                                if msg["role"] == "assistant"), None)
            was_answering_question = last_ai_message and "?" in last_ai_message["content"]
            
            # Get AI response
            with st.spinner("Analyzing symptoms..."):
                # If we've asked 3 questions and user has answered the third one
                if was_answering_question:
                    st.session_state.question_counter += 1
                    
                if st.session_state.question_counter >= 3:
                    # Generate conclusion instead of asking more questions
                    matched_symptom, symptom_data = process_symptom(user_input)
                    if matched_symptom and symptom_data:
                        ai_response = f"Based on your responses, here is my assessment:\n\n"
                        ai_response += f"You may be experiencing {matched_symptom}.\n\n"
                        ai_response += f"Recommendation: {symptom_data['triage_recommendations']['doctor_visit']}\n\n"
                        ai_response += f"Safe advice: {symptom_data['safe_advice']}"
                        
                        # Reset counter for next conversation
                        st.session_state.question_counter = 0
                else:
                    # Normal AI response
                    ai_response = get_langchain_response(user_input)
                    
            # Add AI response to history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
    else:
        st.warning("Please enter your OpenAI API key to use the AI Chat mode.")

    # Display all messages in the container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
            
    # If we've asked enough questions, show a helpful note
    if st.session_state.question_counter >= 3:
        st.success("The assessment is complete. You should now see a recommendation above.")
        
    #Add clarification about what's happening behind the scenes
    with st.expander("How this works"):
        st.write("""
        1. Your symptom description is matched to our medical database
        2. The AI asks up to 3 key questions from our curated list for that symptom
        3. After these questions, the AI provides:
           - Any high-risk conditions to be aware of
           - A triage recommendation (emergency/doctor visit/home care)
           - Safe advice based on medical guidelines
        4. All recommendations come directly from our verified medical database
        """)

# In the Analytics tab
with tab_symptom_analytics:
    st.header("Symptom Analytics")
    
    st.info("Generate charts and insights from the symptom database")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Triage Distribution", "Symptom Complexity", "Conditions to Flag"]
    )
    
    if st.button("Generate Chart"):
        with st.spinner("Creating chart..."):
            create_and_display_chart(symptom_guidelines, chart_type)
                                  
                
# In your NLP Analytics tab
with tab_nlp_analytics:
    st.header("NLP Analytics")
    
    st.info("""
    This tab provides visualizations of the NLP embeddings and semantic relationships 
    between symptoms in our database. These visualizations help understand how the 
    AI interprets and relates different medical symptoms.
    """)
    
    # Initialize session state for embeddings
    if "embeddings_dict" not in st.session_state:
        st.session_state.embeddings_dict = None
    
    # Button to generate/regenerate embeddings
    if st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings for all symptoms... This may take a minute."):
            st.session_state.embeddings_dict = get_all_symptom_embeddings(symptom_guidelines)
            st.success("Embeddings generated successfully!")
    
    # Only show visualization options if embeddings are generated
    if st.session_state.embeddings_dict:
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Symptom Embeddings Map", "Cosine Similarity Heatmap", "Top Similar Symptoms", 
            "Symptom Relationship Network", "Embedding Dimension Importance", "Triage Severity Distribution"]
        )
        
        if analysis_type == "Symptom Embeddings Map":
            st.write("This visualization shows how symptoms are related to each other in the semantic space.")
            if st.button("Generate Embedding Map"):
                with st.spinner("Creating embedding visualization..."):
                    filepath = visualize_embeddings(st.session_state.embeddings_dict)
                    st.success(f"Visualization saved to {filepath}")
        
        elif analysis_type == "Cosine Similarity Heatmap":
            st.write("This heatmap shows how similar each symptom is to every other symptom.")
            if st.button("Generate Heatmap"):
                with st.spinner("Creating similarity heatmap..."):
                    fig, filepath = create_cosine_similarity_heatmap(st.session_state.embeddings_dict)
                    st.pyplot(fig) # draw the heatmap in Streamlit
                    st.success(f"Heatmap saved to {filepath}")
        elif analysis_type == "Top Similar Symptoms":
            st.write("This chart shows which symptoms are most similar to your selected symptom.")
            selected_symptom = st.selectbox("Select a symptom", list(st.session_state.embeddings_dict.keys()), key="symptom_selector")
            top_n = st.slider("Number of similar symptoms to show", 3, 10, 5, key="top_n_slider")
            
            # Add a unique key to the button
            if st.button("Generate Chart", key="similar_symptoms_button"):
                with st.spinner("Creating similarity chart..."):
                    filepath = create_top_similar_symptoms_chart(selected_symptom, st.session_state.embeddings_dict, top_n)
                    st.success(f"Chart saved to {filepath}")
        elif analysis_type == "Symptom Relationship Network":
            st.write("This network graph shows how symptoms are related, with connections representing high similarity.")
            similarity_threshold = st.slider("Similarity threshold", 0.5, 0.9, 0.7, 0.05, key="network_threshold_slider")
            
            if st.button("Generate Network Graph", key="network_graph_button"):
                with st.spinner("Creating network visualization..."):
                    # You'll need to make sure this function is defined in your NLP_Visualization module
                    filepath = create_symptom_network(st.session_state.embeddings_dict, similarity_threshold)
                    st.success(f"Network graph saved to {filepath}")
        elif analysis_type == "Embedding Dimension Importance":
            st.write("""
            This visualization analyzes which dimensions in the embedding space capture the most 
            information about the symptoms. It uses Principal Component Analysis (PCA) to identify
            the most important dimensions.
            """)
            
            if st.button("Generate Dimension Analysis", key="dimension_analysis_button"):
                with st.spinner("Analyzing embedding dimensions..."):
                    filepaths = analyze_embedding_dimensions(st.session_state.embeddings_dict)
                    st.success(f"Analysis complete! Charts saved to {filepaths[0]} and {filepaths[1]}")
                    
                    # Show additional explanation
                    st.info("""
                    **Interpretation:** 
                    - The first chart shows how much variance each principal component explains
                    - The second chart shows how symptoms are distributed in the 2D space of the first two components
                    - Symptoms that are close in this space are semantically similar
                    - The axes represent the most important dimensions of variation in the embedding space
                    """)
        elif analysis_type == "Triage Severity Distribution":
            st.write("""
            This visualization shows how symptoms in the database are distributed across different 
            triage severity levels (Emergency, Doctor Visit, Home Care), and provides examples 
            of symptoms in each category.
            """)
            
            if st.button("Generate Triage Distribution", key="triage_distribution_button"):
                with st.spinner("Analyzing triage severity distribution..."):
                    filepath = create_triage_severity_distribution(symptom_guidelines)
                    st.success(f"Analysis complete! Chart saved to {filepath}")
                    
                    # Show additional explanation
                    st.info("""
                    **Interpretation:** 
                    - The pie chart shows the proportion of symptoms falling into each triage category
                    - The bar chart provides examples of specific symptoms in each category
                    - This helps understand the overall severity distribution in your symptom database
                    """)
    else:
        st.warning("Please generate embeddings first by clicking the 'Generate Embeddings' button above.")
        

st.divider()
st.caption("""
    **Medical Disclaimer**: This app provides general information and is not a substitute for professional medical advice. 
    Always consult a qualified healthcare provider for medical concerns. In case of emergency, call 911 or your local 
    emergency number immediately.
""")
