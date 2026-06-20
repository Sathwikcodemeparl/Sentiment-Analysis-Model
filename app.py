import gradio as gr
import joblib
import re
import pandas as pd
import os
from flask import Flask

print("🧠 Loading local Machine Learning Sentiment Engine...")
try:
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    ml_engine_ready = True
    print("✅ System Core Loaded Successfully!")
except FileNotFoundError:
    ml_engine_ready = False
    print("❌ Error: Missing weights. Run training.py via your control hub first.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def process_and_beautify(user_input):
    # Fallback data structure if text input is blank
    empty_df = pd.DataFrame(columns=["Extracted Word/Phrase", "Matrix Index", "Importance Score"])
    if not user_input.strip() or not ml_engine_ready:
        return (
            "Awaiting Input...", 
            "### ⏳ Waiting for Text Input\nSubmit a sentence above to initialize the data science pipeline.", 
            empty_df, 
            empty_df
        )

    # Stage 1: Clean Text
    cleaned = clean_text(user_input)
    
    # Stage 2 & 3: Run Vectorizer & Classifier
    input_vector = vectorizer.transform([cleaned])
    prediction = sentiment_model.predict(input_vector)[0]
    
    # Map calculated classification zone to a user-friendly label
    if prediction == "positive":
        sentiment_label = "😊 Happy & Positive Vibe"
    else:
        sentiment_label = "😟 Frustrated & Negative Vibe"
    
    # Extract coordinates from the sparse matrix
    non_zero_coords = input_vector.tocoo()
    feature_names = vectorizer.get_feature_names_out()
    words_in_input = [word for word in cleaned.split() if word in vectorizer.vocabulary_]
    
    # Build clean Markdown Storyboard Card
    analysis_markdown = f"""
    ### 🧱 STAGE 1: TEXT SANITIZATION
    > **Action:** Strips punctuation, numbers, and forces lowercasing to normalize input data shapes.
    * **Raw String:** *"{user_input}"*
    * **Cleaned Base:** `{cleaned}`
    
    ---
    ### 🔤 STAGE 2: VOCABULARY MATCHING
    > **Action:** Scans the text and maps it directly against your  dataset features.
    * **Isolated Key Tokens:** `{words_in_input if words_in_input else "None found in dictionary"}`
    
    ---
    ### 📐 STAGE 3: CLASSIFIER BOUNDARY
    > **Action:** Feeds the resulting coordinate layout into your LinearSVC to see which classification zone it lands in.
    * **Calculated Outcome:** **{sentiment_label}**
    """

    # Format the Table and Graph Data
    pipeline_data = []
    for col in non_zero_coords.col:
        pipeline_data.append({
            "Extracted Word/Phrase": feature_names[col],
            "Matrix Index": int(col),
            "Importance Score": round(float(input_vector[0, col]), 4)
        })
        
    df_pipeline = pd.DataFrame(pipeline_data)
    
    # Safely sort tokens by weight importance
    if not df_pipeline.empty:
        df_pipeline = df_pipeline.sort_values(by="Importance Score", ascending=False).reset_index(drop=True)
    else:
        df_pipeline = pd.DataFrame([{"Extracted Word/Phrase": "None", "Matrix Index": 0, "Importance Score": 0.0}])

    # Return elements directly to the layout hooks
    return sentiment_label, analysis_markdown, df_pipeline, df_pipeline

# ──► CONSTRUCT INTERACTIVE LAYOUT BLOCKS ◄──
with gr.Blocks(title="📊 NLP Vectorization Dashboard") as iface:
    gr.Markdown("# 📊 Text Analytics Pipeline Dashboard")
    gr.Markdown("Watch how machine learning algorithms systematically clean, tokenize, and transform your language into numerical vector spaces.")
    
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                placeholder="Type your feedback message here (e.g., 'The icebreaker games were absolutely fantastic and well organized')...", 
                label="User Feedback Input", 
                lines=2
            )
            submit_btn = gr.Button("Execute Analysis", variant="primary")
            
        with gr.Column(scale=1):
            emotion_output = gr.Label(label="Algorithmic Mood Classification")

    gr.Markdown("---")
    gr.Markdown("## ⚙️ Core Processing Metrics")
    
    with gr.Row():
        with gr.Column(scale=1):
            analysis_card = gr.Markdown(value="### ⏳ Waiting for Text Input\nSubmit a sentence above to initialize the data science pipeline.")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Vector Features Matrix")
            matrix_table = gr.Dataframe(
                headers=["Extracted Word/Phrase", "Matrix Index", "Importance Score"],
                datatype=["str", "number", "number"],
                interactive=False
            )
            
            # Chart structural container defined out of function scopes for Gradio 6 compatibility
            chart_output = gr.BarPlot(
                x="Extracted Word/Phrase",
                y="Importance Score",
                title="🎯 Keyword Weight Distribution (TF-IDF)",
                tooltip=["Matrix Index", "Importance Score"],
                y_title="Statistical Significance",
                x_title="Tokens"
            )

    # Event Triggers
    submit_btn.click(
        fn=process_and_beautify, 
        inputs=user_input, 
        outputs=[emotion_output, analysis_card, matrix_table, chart_output]
    )
    user_input.submit(
        fn=process_and_beautify, 
        inputs=user_input, 
        outputs=[emotion_output, analysis_card, matrix_table, chart_output]
    )


# 🚨 VERCEL SERVERLESS PRODUCTION ROUTING LAYER
from fastapi import FastAPI

# 1. Initialize a FastAPI app
app = FastAPI()

# 2. Mount your Gradio interface onto the root path safely
app = gr.mount_gradio_app(app, iface, path="/")

# 🚨 THE FIX: Only call launch() if running directly on your laptop
if __name__ == "__main__":
    # This block executes locally via interactive_test.py
    # Vercel reads the 'app' variable above and completely ignores this block
    iface.launch(debug=True)