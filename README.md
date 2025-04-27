
# Medical Symptom Advisor

An AI-powered, explainable, high-risk healthcare decision support system that assists users in early symptom triage through semantic understanding, conversational reasoning, real-time external data enrichment, and visual analytics.

---

## Features

- Semantic symptom matching using FAISS vector search and OpenAI embeddings
- Multi-turn clarification dialogs powered by Chain-of-Thought reasoning
- Evidence-supported triage recommendations (Emergency, Doctor Visit, Monitor at Home)
- Real-time integration with OpenFDA and PubMed for external medical data validation
- Interactive visualization tools: t-SNE clustering, cosine similarity heatmaps, and symptom networks
- Streamlit-based web application interface
- Secure environment variable management (.env)

---

## Project Structure

- `frontend_app.py` — Streamlit app frontend for user interaction
- `symptom_matcher.py` — Semantic matching and LangChain initialization
- `triage_services.py` — Triage question sequencing and final recommendation generation
- `symptom_analytics_charts.py` — Triage distribution visualization
- `visualization.py` — Embedding and symptom relationship visualization
- `advisor_enrichment.py` — External enrichment using OpenFDA and PubMed APIs
- `data/combined_symptom_database.json` — Structured symptom database

---

## Running the Application

```bash
streamlit run frontend_app.py
```

Access the app at: http://localhost:8501/

---

## Future Enhancements

- Expand symptom coverage and rare disease database
- Integrate real-time clinical feeds (WHO, CDC)
- Improve multi-lingual support and mobile optimization
- Conduct clinical validation studies for real-world deployment

---

## License

This project is licensed under the MIT License.

---

## Contact

Developed by **Satish Swarnkar**  
University of Texas at Austin  
Email: swarnkar.satish@gmail.com

---
