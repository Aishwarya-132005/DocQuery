import os
import math
import shutil
import streamlit as st

import json

from ingest import process_pdf
from rag import ask_question, load_llm, load_judge_llm
from config import UPLOAD_DIR

st.set_page_config(
    page_title="PDF RAG using LangChain + Hugging Face",
    layout="wide"
)

# --- Custom Styling for Professional Look ---
st.markdown("""
<style>
    /* Sleek gradient background */
    .stApp {
        background: linear-gradient(135deg, #0A0F24 0%, #121C38 100%);
        color: #F8FAFC;
    }
    
    /* Professional modern inputs */
    .stChatInputContainer {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 36, 0.95);
        border-right: 1px solid #1E293B;
    }
    
    /* Elegant metric highlights */
    [data-testid="stMetricValue"] {
        color: #38BDF8;
    }
    
    /* Remove default generic Streamlit top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Document Intelligence")
st.markdown("#### *Extract insights using advanced RAG and LLM Evaluation.*")

UPLOAD_DIR.mkdir(exist_ok=True)

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# RAGAS Per-Question Evaluation


def safe_metric_value(value):
    """Safely extract a numeric metric value, handling lists, NaN, and None."""
    if isinstance(value, list):
        value = value[0] if len(value) > 0 else float('nan')
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


LLM_JUDGE_PROMPT = """You are an impartial judge evaluating a Retrieval-Augmented Generation (RAG) system. 
You will be provided with a User Question, a Generated Answer, and the Retrieved Context.

Please evaluate the Generated Answer based on the following four metrics, scoring each from 0.0 to 1.0 (where 1.0 is the best):
1. Faithfulness: Is the answer derived entirely from the retrieved context without making up information?
2. Answer Relevancy: How directly does the answer address the user's question?
3. Context Precision: Are the retrieved context chunks relevant to the user's question?
4. Context Recall: Does the retrieved context contain all the information needed to answer the question?

Provide a brief explanation for each score.

CRITICAL: Your output MUST be strictly valid JSON. Do not include markdown code blocks. Escape any double quotes inside your explanations using \".
Return the results EXACTLY in the following JSON format (replace the word SCORE with a float like 0.8, and EXPLANATION with your reasoning):
{{
    "faithfulness": {{"score": SCORE, "explanation": "EXPLANATION"}},
    "answer_relevancy": {{"score": SCORE, "explanation": "EXPLANATION"}},
    "context_precision": {{"score": SCORE, "explanation": "EXPLANATION"}},
    "context_recall": {{"score": SCORE, "explanation": "EXPLANATION"}}
}}

User Question: {question}
Generated Answer: {answer}
Retrieved Context: {context}
"""

def evaluate_single_question(question, answer, context_docs):
    """Evaluate a single Q&A pair using an LLM-as-a-Judge approach."""
    contexts = "\\n\\n".join([doc.page_content for doc in context_docs])
    
    try:
        llm = load_judge_llm()
        prompt = LLM_JUDGE_PROMPT.format(question=question, answer=answer, context=contexts)
        
        # Invoke LLM
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON
        """It checks whether an object has a particular attribute or not.
        Returns True → Attribute exists.     
        Returns False → Attribute does not exist."""
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx+1]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response.")
        
        return {
            "faithfulness": safe_metric_value(result.get("faithfulness", {}).get("score", float('nan'))),
            "answer_relevancy": safe_metric_value(result.get("answer_relevancy", {}).get("score", float('nan'))),
            "context_precision": safe_metric_value(result.get("context_precision", {}).get("score", float('nan'))),
            "context_recall": safe_metric_value(result.get("context_recall", {}).get("score", float('nan'))),
            "explanations": {
                "faithfulness": result.get("faithfulness", {}).get("explanation", ""),
                "answer_relevancy": result.get("answer_relevancy", {}).get("explanation", ""),
                "context_precision": result.get("context_precision", {}).get("explanation", ""),
                "context_recall": result.get("context_recall", {}).get("explanation", "")
            }
        }
    except Exception as e:
        st.warning(f"LLM Judge evaluation encountered an error: {e}")
        return {
            "faithfulness": float('nan'),
            "answer_relevancy": float('nan'),
            "context_precision": float('nan'),
            "context_recall": float('nan'),
            "explanations": {}
        }


def format_metric(value):
    """Format a metric value for display, showing N/A for NaN."""
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


with st.sidebar:
    st.title("Control Center")
    st.markdown("---")
    
    st.subheader("Settings")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []

    if st.button("Reset Database", use_container_width=True):
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")
        st.session_state.processed = False
        st.success("Vector DB deleted.")

# --- Document Management (Main Page) ---
with st.expander("Document Management", expanded=not st.session_state.processed):
    uploaded_file = st.file_uploader(
        "Select a PDF file to begin processing",
        type=["pdf"]
    )
    
    if uploaded_file is not None:
        save_path = UPLOAD_DIR / uploaded_file.name
    
        # Check if we have already processed this exact file
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.processed = False
    
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
        if not st.session_state.processed:
            with st.spinner("Creating Vector Database..."):
                total_chunks = process_pdf(save_path)
            st.success("PDF processed successfully.")
            st.info(f"Total Chunks : {total_chunks}")
            st.session_state.processed = True
        else:
            st.success("PDF uploaded and processed successfully.")
            
st.markdown("---")

# --- Chat History Display ---
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="human"):
            st.markdown(chat["question"])
            
        with st.chat_message("assistant", avatar="assistant"):
            st.markdown(chat["answer"])
            
            # Show Evaluation Metrics in an expander to keep UI clean
            if "metrics" in chat and chat["metrics"]:
                with st.expander("Evaluation Metrics (RAGAS)"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Faithfulness", format_metric(chat['metrics']['faithfulness']))
                    c2.metric("Answer Relevancy", format_metric(chat['metrics']['answer_relevancy']))
                    c3.metric("Context Precision", format_metric(chat['metrics']['context_precision']))
                    c4.metric("Context Recall", format_metric(chat['metrics']['context_recall']))
                    
                    st.markdown("**Judge's Reasoning:**")
                    st.info(f"**Faithfulness:** {chat['metrics'].get('explanations', {}).get('faithfulness', 'N/A')}")
                    st.info(f"**Answer Relevancy:** {chat['metrics'].get('explanations', {}).get('answer_relevancy', 'N/A')}")
                    st.info(f"**Context Precision:** {chat['metrics'].get('explanations', {}).get('context_precision', 'N/A')}")
                    st.info(f"**Context Recall:** {chat['metrics'].get('explanations', {}).get('context_recall', 'N/A')}")
            
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(chat["documents"], start=1):
                    st.markdown(f"**Chunk {i}** (Page {doc.metadata.get('page', 'N/A')})")
                    st.caption(doc.page_content)

# --- Chat Input ---
if question := st.chat_input("Ask a question about your PDF..."):
    
    if not st.session_state.processed:
        st.warning("Please upload and process a PDF first.")
    else:
        # Display user message immediately
        with st.chat_message("user", avatar="human"):
            st.markdown(question)
            
        # Display assistant thinking/response
        with st.chat_message("assistant", avatar="assistant"):
            with st.spinner("Generating answer..."):
                response = ask_question(question)
                answer = response["answer"]
                documents = response["context"]
                
            st.markdown(answer)
            
            with st.spinner("Evaluating response..."):
                metrics = evaluate_single_question(question, answer, documents)
                
            with st.expander("Evaluation Metrics (RAGAS)"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness", format_metric(metrics['faithfulness']))
                c2.metric("Answer Relevancy", format_metric(metrics['answer_relevancy']))
                c3.metric("Context Precision", format_metric(metrics['context_precision']))
                c4.metric("Context Recall", format_metric(metrics['context_recall']))
                
                st.markdown("**Judge's Reasoning:**")
                st.info(f"**Faithfulness:** {metrics.get('explanations', {}).get('faithfulness', 'N/A')}")
                st.info(f"**Answer Relevancy:** {metrics.get('explanations', {}).get('answer_relevancy', 'N/A')}")
                st.info(f"**Context Precision:** {metrics.get('explanations', {}).get('context_precision', 'N/A')}")
                st.info(f"**Context Recall:** {metrics.get('explanations', {}).get('context_recall', 'N/A')}")
            
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(documents, start=1):
                    st.markdown(f"**Chunk {i}** (Page {doc.metadata.get('page', 'N/A')})")
                    st.caption(doc.page_content)
        
        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "documents": documents,
            "metrics": metrics,
        })
