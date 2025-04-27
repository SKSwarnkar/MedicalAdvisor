# visualization.py
#
# Medical Symptom Advisor - Advanced Visualization Module
#
# This script handles the visualization of semantic relationships between symptoms.
#
# Main Functionalities:
# - Generate t-SNE projections of symptom embeddings for visual clustering.
# - Create cosine similarity heatmaps between symptoms.
# - Build network graphs showing semantic proximity between symptoms.
# - Compute symptom embeddings dynamically using OpenAI APIs.
# - Cache computation results to optimize performance in Streamlit.
#
# Core Components:
# - OpenAI embeddings for symptom semantic understanding.
# - Cosine similarity calculations between symptom vectors.
# - Dimensionality reduction using t-SNE.
# - Graph-based visualizations using NetworkX and Matplotlib.
#
# Libraries Used:
# - streamlit
# - openai
# - dotenv
# - numpy
# - matplotlib.pyplot
# - seaborn
# - sklearn.manifold (TSNE)
# - sklearn.metrics (cosine_similarity)
# - datetime
#
# Author: Satish Swarnkar
# University of Texas at Austin


import streamlit as st
import os
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable if not already set
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data(ttl=3600)  # Cache results for 1 hour

# Functions for embeddings and analytics
def get_embedding_for_symptom(symptom, symptom_data=None):
    """
    Get embedding vector for a symptom using OpenAI embeddings.
    """
    # Import your existing OpenAI embeddings model
    from symptom_matcher import embeddings

    # If embeddings haven't been initialized yet
    if embeddings is None:
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a rich text representation of the symptom
    if symptom_data:
        text = f"Symptom: {symptom}. "
        if symptom_data.get('clarifying_questions', []):
            text += f"Description: Symptoms include {symptom_data.get('clarifying_questions', [])[0]} "
        if symptom_data.get('conditions_to_flag', []):
            text += f"and possible {symptom_data.get('conditions_to_flag', [])[0]}."
    else:
        text = f"Medical symptom: {symptom}"
    
    # Get embedding from OpenAI
    embedding_list = embeddings.embed_query(text)
    
    # Convert to numpy array for easier manipulation
    return np.array(embedding_list)

def get_all_symptom_embeddings(symptom_guidelines):
    """
    Get embeddings for all symptoms in the database
    """
    
    embeddings_dict = {}
    
    for symptom, data in symptom_guidelines.items():
        embedding = get_embedding_for_symptom(symptom, data)
        embeddings_dict[symptom] = embedding
        
    return embeddings_dict

def create_cosine_similarity_heatmap(embeddings_dict):
    """Create a heatmap of cosine similarities between symptom embeddings"""
    
    # Get list of symptoms and their embeddings
    symptoms = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, 
        annot=False,
        xticklabels=symptoms,
        yticklabels=symptoms,
        cmap='viridis',
        vmin=0, 
        vmax=1
    )
    plt.title('Cosine Similarity Between Symptom Embeddings')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Display in Streamlit
    # st.pyplot(fig)
    
    # Save the visualization
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"outputs/cosine_similarity_heatmap_{timestamp}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return fig, filepath

def create_top_similar_symptoms_chart(symptom, embeddings_dict, top_n=5):
    """Create a bar chart showing top N most similar symptoms to the selected one"""
    
    # Get embedding for the selected symptom
    selected_embedding = embeddings_dict[symptom]
    
    # Calculate similarity with all other symptoms
    similarities = {}
    for other_symptom, embedding in embeddings_dict.items():
        if other_symptom != symptom:
            sim = cosine_similarity([selected_embedding], [embedding])[0][0]
            similarities[other_symptom] = sim
    
    # Get top N similar symptoms
    top_symptoms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    symptoms_names = [s[0] for s in top_symptoms]
    similarity_scores = [s[1] for s in top_symptoms]
    
    y_pos = np.arange(len(symptoms_names))
    ax.barh(y_pos, similarity_scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(symptoms_names)
    ax.invert_yaxis()  # Highest similarity at the top
    ax.set_xlabel('Cosine Similarity')
    ax.set_title(f'Top {top_n} Symptoms Similar to "{symptom}"')
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"outputs/top_similar_to_{symptom}_{timestamp}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return filepath

def visualize_embeddings(embeddings_dict):
    """Create and display a t-SNE visualization of symptom embeddings"""
    
    # Get list of symptoms and their embeddings
    symptoms = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))
    
    # Check number of samples
    n_samples = len(symptoms)
    
    # Adjust perplexity based on number of samples
    if n_samples < 5:
        st.warning(f"Only {n_samples} symptoms found. Need at least 5 for good visualization.")
        perplexity = 1  # Set to minimum
    elif n_samples < 10:
        perplexity = 2
    elif n_samples < 30:
        perplexity = 5
    else:
        perplexity = 30  # Default for larger datasets
    
    # Apply t-SNE to reduce dimensions to 2D with adjusted perplexity
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    # Apply t-SNE to reduce dimensions to 2D
    # tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each symptom as a point
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add labels for each point
    for i, symptom in enumerate(symptoms):
        ax.annotate(symptom, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title('Symptom Embeddings Visualization (t-SNE)')
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Save the visualization
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"outputs/symptom_embeddings_{timestamp}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return filepath

def create_symptom_network(embeddings_dict, similarity_threshold=0.7):
    """Create a network graph visualization of related symptoms"""
    import networkx as nx
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes (symptoms)
    symptoms = list(embeddings_dict.keys())
    for symptom in symptoms:
        G.add_node(symptom)
    
    # Add edges (relationships) between symptoms with high similarity
    for i, symptom1 in enumerate(symptoms):
        embedding1 = embeddings_dict[symptom1]
        for symptom2 in symptoms[i+1:]:
            embedding2 = embeddings_dict[symptom2]
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Add edge if similarity is above threshold
            if similarity > similarity_threshold:
                G.add_edge(symptom1, symptom2, weight=similarity)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw edges with width based on similarity
    edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Symptom Relationship Network")
    plt.axis('off')
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Save the visualization
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"outputs/symptom_network_{timestamp}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return filepath

def analyze_embedding_dimensions(embeddings_dict):
    """Analyze which dimensions in embeddings carry the most information using PCA"""
    from sklearn.decomposition import PCA
    import pandas as pd
    
    # Get embeddings as array
    embeddings = np.array(list(embeddings_dict.values()))
    symptom_names = list(embeddings_dict.keys())
    
    # Apply PCA
    n_components = min(10, embeddings.shape[1])  # Take top 10 components or less
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    # Create dataframe for variance explained
    variance_explained = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_explained)
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Variance explained by each component
    ax1.bar(range(1, n_components+1), variance_explained)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained by Each Principal Component')
    ax1.set_xticks(range(1, n_components+1))
    
    # Plot 2: Cumulative variance explained
    ax2.plot(range(1, n_components+1), cumulative_variance, marker='o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_xticks(range(1, n_components+1))
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.5)  # 90% reference line
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Project symptoms onto the first two principal components
    pca_result = pca.transform(embeddings)
    
    # Create second figure showing symptom distribution in PC space
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    
    # Plot symptoms in PC space
    ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    
    # Add labels for each point
    for i, symptom in enumerate(symptom_names):
        ax3.annotate(symptom, (pca_result[i, 0], pca_result[i, 1]))
    
    ax3.set_xlabel(f'Principal Component 1 ({variance_explained[0]:.1f}%)')
    ax3.set_ylabel(f'Principal Component 2 ({variance_explained[1]:.1f}%)')
    ax3.set_title('Symptoms Projected onto First Two Principal Components')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Display in Streamlit
    st.pyplot(fig)
    st.pyplot(fig2)
    
    # Save the visualizations
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath1 = f"outputs/pca_variance_explained_{timestamp}.png"
    filepath2 = f"outputs/pca_symptom_projection_{timestamp}.png"
    
    fig.savefig(filepath1, dpi=300, bbox_inches='tight')
    fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
    
    return filepath1, filepath2

def create_triage_severity_distribution(symptom_guidelines):
    """
    Create a visualization showing the distribution of symptoms by triage severity
    """
    # Categorize symptoms by triage severity
    triage_categories = {
        "Emergency": [],
        "Doctor Visit": [],
        "Home Care": []
    }
    
    # Analyze each symptom's triage recommendations to categorize it
    for symptom, data in symptom_guidelines.items():
        # Check emergency recommendation for urgent keywords
        emergency_text = data['triage_recommendations']['emergency'].lower()
        doctor_text = data['triage_recommendations']['doctor_visit'].lower()
        home_text = data['triage_recommendations']['monitor_at_home'].lower()
        
        # Determine primary triage category based on text analysis
        if any(word in emergency_text for word in ['immediate', 'emergency', 'urgent', 'call 911']):
            triage_categories["Emergency"].append(symptom)
        elif len(emergency_text) > len(doctor_text) and len(emergency_text) > len(home_text):
            triage_categories["Emergency"].append(symptom)
        elif any(word in doctor_text for word in ['consult', 'see a doctor', 'physician', 'healthcare provider']):
            triage_categories["Doctor Visit"].append(symptom)
        elif len(doctor_text) > len(home_text):
            triage_categories["Doctor Visit"].append(symptom)
        else:
            triage_categories["Home Care"].append(symptom)
    
    # Count the number of symptoms in each category
    category_counts = {category: len(symptoms) for category, symptoms in triage_categories.items()}
    
    # Create the pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Pie chart of category distributions
    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  # explode the emergency slice
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Distribution of Symptoms by Triage Severity')
    
    # Plot 2: Bar chart showing examples of symptoms in each category
    categories = []
    symptom_names = []
    
    # Get up to 5 examples from each category
    for category, symptoms in triage_categories.items():
        for symptom in symptoms[:5]:  # take first 5 or fewer
            categories.append(category)
            symptom_names.append(symptom)
    
    # Create positions for the bars
    y_pos = np.arange(len(symptom_names))
    
    # Create colors for the bars based on category
    bar_colors = []
    for category in categories:
        if category == "Emergency":
            bar_colors.append(colors[0])
        elif category == "Doctor Visit":
            bar_colors.append(colors[1])
        else:
            bar_colors.append(colors[2])
    
    # Create horizontal bar chart
    ax2.barh(y_pos, [1] * len(symptom_names), color=bar_colors, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(symptom_names)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Examples of Symptoms in Each Category')
    ax2.set_title('Example Symptoms by Triage Category')
    ax2.set_xticks([])  # Hide x-axis ticks since they're not meaningful
    
    # Add category labels as a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=4, label='Emergency'),
        Line2D([0], [0], color=colors[1], lw=4, label='Doctor Visit'),
        Line2D([0], [0], color=colors[2], lw=4, label='Home Care')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Save the visualization
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"outputs/triage_severity_distribution_{timestamp}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return filepath