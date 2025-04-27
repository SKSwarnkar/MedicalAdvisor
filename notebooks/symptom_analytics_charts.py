# symptom_analytics_charts.py
#
# Medical Symptom Advisor - Symptom Analytics and Visualization Module
#
# This script creates and displays analytical visualizations related to symptom triage distribution.
#
# Main Functionalities:
# - Analyze the distribution of triage recommendations across symptoms.
# - Create pie charts showing proportions of Emergency, Doctor Visit, and Home Care cases.
# - Save generated charts to an outputs directory.
# - Display visualizations directly inside the Streamlit app.
#
# Core Components:
# - Read symptom triage data and categorize by urgency levels.
# - Generate Matplotlib/Seaborn-based pie charts for visual summaries.
# - Streamlit integration for direct chart rendering in the UI.
#
# Libraries Used:
# - streamlit
# - matplotlib.pyplot
# - pandas
# - seaborn
# - os, datetime
#
# Author: Satish Swarnkar
# University of Texas at Austin


import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from datetime import datetime

# Create an outputs directory if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")

def create_and_display_chart(symptom_data, chart_type):
    """Create, display, and optionally save a chart analyzing symptom triage distribution.
    
    Parameters:
    - symptom_data (dict): Symptom database entries with triage recommendations.
    - chart_type (str): Type of chart to generate (currently supports 'Triage Distribution').
    """    
    
    if chart_type == "Triage Distribution":
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count occurrences of each triage recommendation
        triage_counts = {
            "Emergency": 0,
            "Doctor Visit": 0,
            "Home Care": 0
        }
        
        for symptom, data in symptom_data.items():
            # Categorize based on keywords in recommendations
            if "immediate" in data["triage_recommendations"]["emergency"].lower():
                triage_counts["Emergency"] += 1
            elif "doctor" in data["triage_recommendations"]["doctor_visit"].lower():
                triage_counts["Doctor Visit"] += 1
            else:
                triage_counts["Home Care"] += 1
        
        # Create pie chart
        labels = triage_counts.keys()
        sizes = triage_counts.values()
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Distribution of Triage Recommendations')
        
        # Display the chart directly in Streamlit
        st.pyplot(fig)
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/triage_distribution_{timestamp}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Show success message with file path
        st.success(f"Chart saved to {filepath}")
        
    elif chart_type == "Symptom Complexity":
        # Create dataframe of symptom complexity
        data = []
        for symptom, details in symptom_data.items():
            data.append({
                'Symptom': symptom,
                'Questions Count': len(details['clarifying_questions'])
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Questions Count', ascending=False)
        
        # Create the bar chart using Streamlit's native chart function
        st.bar_chart(df.set_index('Symptom'))
        
        # Also save using matplotlib
        fig, ax = plt.subplots(figsize=(12, 8))
        df.plot.bar(x='Symptom', y='Questions Count', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/symptom_complexity_{timestamp}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        st.success(f"Chart saved to {filepath}")
    elif chart_type == "Conditions to Flag":
        # Count conditions to flag across all symptoms
        condition_counts = {}
        
        # Extract all conditions from all symptoms
        for symptom, details in symptom_data.items():
            for condition in details.get('conditions_to_flag', []):
                if condition in condition_counts:
                    condition_counts[condition] += 1
                else:
                    condition_counts[condition] = 1
        
        # Sort conditions by frequency (most common first)
        sorted_conditions = dict(sorted(condition_counts.items(), 
                                        key=lambda item: item[1], 
                                        reverse=True))
        
        # Take top 15 for readability if there are many
        if len(sorted_conditions) > 15:
            top_conditions = {k: sorted_conditions[k] for k in list(sorted_conditions.keys())[:15]}
        else:
            top_conditions = sorted_conditions
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bars
        y_pos = range(len(top_conditions))
        ax.barh(y_pos, list(top_conditions.values()), align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(top_conditions.keys()))
        ax.invert_yaxis()  # Put the top condition at the top
        
        ax.set_xlabel('Number of Symptoms')
        ax.set_title('Most Common Conditions to Flag Across Symptoms')
        
        # Add count labels on bars
        for i, v in enumerate(top_conditions.values()):
            ax.text(v + 0.1, i, str(v), va='center')
        
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"outputs/conditions_to_flag_{timestamp}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        st.success(f"Chart saved to {filepath}")
        return filepath