import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from servicenow_manager import submit_to_servicenow
st.set_page_config(page_title="IntelliTicket: AI Ticket Automation", layout="centered", page_icon="🎫")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("assets/style.css")


@st.cache_resource
def load_model():
    model_path = r"output\\best_model"
    device = 0 if torch.cuda.is_available() else -1     # Autodetect GPU

    clf = pipeline("text-classification",
                   model=model_path,
                   tokenizer=model_path,
                   top_k=None,
                   device=device)
    return clf


try:
    clf = load_model()
except Exception as e:
    st.error(f"Failed to load model from path. Error: {e}")
    st.stop()

with st.sidebar:
    st.title("System Info")
    expander = st.expander("**Core Logic**", expanded=True)
    expander.write("""
    - **Model**: DistilBERT-Base-Uncased
    - **Integration**: ServiceNow REST API (Table API)
    - **Confidence Threshold**: 80% (Automatic routing vs. Manual review)
    - **Categories**: Billing, Account, Technical, Delivery
    """)

    st.write("---")
    st.markdown("""
    **Categories handled:**
    - 💳 Billing
    - 👤 Account
    - ⚙ Technical Issue
    - 📦 Delivery
    """)
    with st.expander("See Ticket Specifications"):
        st.write("""
        This is an explanation of your machine learning scripts broken down into simple sections, like a "Machine Learning 101" guide.

    ### Part 1: Explaining train.py
    #### 1. Imports
    We start by grabbing specialized toolkits. pandas is for reading spreadsheets, sklearn is for organizing the data, and transformers provides the "brain" (DistilBERT) we are going to use.

    #### 2. Data Loading
    The script reads customer\_support\_tickets.csv. This is our "Textbook" containing thousands of examples of what customers say and which category they belong to.

    #### 3. Label Encoding
    Computers are bad at words but great at numbers. We take categories like "Billing" or "Technical Issue" and turn them into numbers (0, 1, 2, etc.) so the model can do math on them.
    *   **label2id**: Maps "Billing" $\\rightarrow$ 0.
    *   **id2label**: Maps 0 $\\rightarrow$ "Billing" (so we can read the results later).

    #### 4. Data Splitting
    We don't give the student all the answers at once. We hide 20% of the data (test\_size=0.2). The model studies the first 80%, and then we "exam" it using the hidden 20% to see if it actually learned or just memorized.

    #### 5. Tokenization
    Models can't read whole sentences. We use a **Tokenizer** to chop text into small pieces called "tokens".
    *   **Truncation**: If a ticket is too long, we cut it off.
    *   **Padding**: If a ticket is too short, we add "empty space" so every input is the same length.

    #### 6. Model Initialization
    We load distilbert-base-uncased. This is a pre-trained "brain" that already understands English. We are just "fine-tuning" it to understand _your_ specific support tickets.

    #### 7. Training Arguments
    We tell the teacher how to teach:
    *   **Learning Rate**: How fast the student should try to learn.
    *   **Epochs**: How many times the student should read the entire textbook.
    *   **Batch Size**: How many examples to look at before taking a break to update its knowledge.

    #### 8. The Trainer
    The Trainer is the engine that connects the student (Model) with the data and the rules (Arguments). It runs the loop of: _Guess $\\rightarrow$ Check Answer $\\rightarrow$ Learn from Mistake_.

    #### 9. Saving
    Once training is done, we save the "learned knowledge" into a folder. This creates files that the Streamlit app can later load to make predictions.

    ### Part 2: Explaining app.py
    #### 1. UI Config
    st.set\_page\_config sets up the website title and layout so it looks like a professional app rather than a blank page.

    #### 2. Loading the Model
    We use @st.cache\_resource. This is a performance trick: it tells the app, "Load the model once and keep it in memory; don't reload it every time the user clicks a button." We use a pipeline, which is the easiest way to tell the model to take text and give back a label.

    #### 3. User Input
    st.text\_area creates a big box where a user can type their problem. We add a "Classify" button so the model only starts working when the user is ready.

    #### 4. Prediction Logic
    When the button is clicked:
    1.  The app sends the text to the model.
    2.  The model returns a **Category** and a **Score** (Confidence).
    3.  The score is a number between 0 and 1 (like 0.95 for 95%).

    #### 5. Threshold Logic
    This is the most important part of the app's logic.
    *   **If the score is high (above 0.8)**: The app trusts the model and "auto-assigns" the ticket.
    *   **If the score is low (below 0.8)**: The app says, "I'm not sure, a human needs to look at this." This prevents the "Robot" from making confident mistakes.

    #### 6. Visualization
    *   **Metrics**: Big bold numbers for the category.
    *   **Bar Charts**: A visual look at how much the model debated between different categories (e.g., 90% sure it's Billing, 5% sure it's Technical).
        """)

st.title("IntelliTicket: AI Ticket Automation")
st.markdown("### Classify and Submit issues to ServiceNow in one click.")

ticket_text = st.text_area("", height=120, placeholder="Paste the ticket details here...")

# processing
if st.button("Submit", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket description to predict category.")
    else:
        with st.spinner("Analyzing..."):
            results = clf(ticket_text)[0]

            top_prediction = results[0]
            category = top_prediction['label']
            score = top_prediction['score']
            st.divider()

            if score < 0.8:
                st.warning("**Low Confidence Alert**: This ticket requires manual human review. The model is not certain enough to auto-route this request.")
            else:
                with st.spinner("Submitting to ServiceNow..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category", category)
                    with col2:
                        status = "High Confidence" if score >= 0.8 else "Low Confidence"
                        st.metric("Confidence Score", f"{score:.1%}", delta=status)
                    with st.expander("View Detailed Analysis"):
                        df_results = pd.DataFrame(results)
                        df_results.columns = ['Category', 'Confidence']
                        st.bar_chart(df_results.set_index('Category'))
                        st.table(df_results)

                    success, ticket_info = submit_to_servicenow_pysnow(category, ticket_text)
        
                    if success:
                        st.info(f"Ticket Created: **{ticket_info}**")
                    else:
                        st.error(f"Error: {ticket_info}")

if st.button("Clear Text"):
    st.rerun()

footer_html = """
<div class="footer">
    <p>Developed by <a href="https://www.linkedin.com/in/sandeep-jain-9866392b/" target="_blank">Sandeep Jain </a> | <a href="https://github.com/ksandeepjain" target="_blank"> GIT </a></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
