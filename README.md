# DocQuery: PDF RAG with LangChain & RAGAS Evaluation

DocQuery is an end-to-end **Retrieval-Augmented Generation (RAG)** application that enables users to ask natural language questions about PDF documents and receive context-aware, accurate responses. The system combines document retrieval with Large Language Models (LLMs) to generate answers grounded in the uploaded document rather than relying solely on the model's internal knowledge.

A key component of this project is the integration of the **RAGAS (Retrieval-Augmented Generation Assessment)** framework, which evaluates both the retrieval quality and the generated responses using standard RAG evaluation metrics.

---

## Features

* Upload and interactively query PDF documents.
* Automatic PDF parsing and intelligent text chunking.
* Semantic search using FAISS vector database.
* Context-aware answer generation using open-source LLMs.
* Built-in RAGAS evaluation pipeline for measuring retrieval and generation quality.
* Simple and interactive Streamlit-based user interface.

---

## RAGAS Evaluation Metrics

The project evaluates the RAG pipeline using the following metrics:

* **Faithfulness** – Measures whether the generated answer is supported by the retrieved context.
* **Answer Relevancy** – Evaluates how well the generated response addresses the user's question.
* **Context Precision** – Measures whether the retrieved chunks are relevant and ranked effectively.
* **Context Recall** – Determines whether the retrieved context contains all the information required to answer the query.

---

## Tech Stack

**Frontend**

* Streamlit

**Framework**

* LangChain

**Vector Database**

* FAISS

**Document Loader**

* PyPDFLoader

**Text Chunking**

* RecursiveCharacterTextSplitter

**Embeddings**

* HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

**Large Language Models**

* Llama-3.1-8B-Instruct (Answer Generation)
* Qwen2.5-7B-Instruct (Evaluation)

**Evaluation Framework**

* RAGAS

---

## Workflow

1. Users upload one or more PDF documents through the Streamlit interface.
2. The PDFs are loaded and parsed using **PyPDFLoader**.
3. Documents are split into overlapping text chunks using **RecursiveCharacterTextSplitter**.
4. Each chunk is converted into vector embeddings using the HuggingFace **all-MiniLM-L6-v2** embedding model.
5. The embeddings are stored in a **FAISS** vector database for efficient semantic retrieval.
6. When a user submits a query, the system retrieves the most relevant document chunks from FAISS.
7. The retrieved context and user query are provided to the **Llama-3.1-8B-Instruct** model to generate a context-grounded answer.
8. The generated responses are evaluated against ground-truth answers using the **RAGAS** framework.
9. Evaluation scores for Faithfulness, Answer Relevancy, Context Precision, and Context Recall are saved to `ragas_results.csv`.

---

## Project Highlights

* End-to-end Retrieval-Augmented Generation pipeline.
* Local semantic search using FAISS.
* Open-source embedding and language models.
* Automated evaluation using RAGAS.
* Modular architecture for easy experimentation with different embedding models, LLMs, and vector databases.
* User-friendly Streamlit interface for document-based question answering.

---
