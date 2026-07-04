from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ==========================================
# Load PDF
# ==========================================

def load_pdf(pdf_path: str):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# ==========================================
# Split Documents
# ==========================================

def clean_text(text):
    """Clean extracted PDF text to remove noise."""
    import re
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_documents(documents):

    # Clean text in each document before splitting
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Filter out very short chunks that are just noise
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    return chunks


# ==========================================
# Embedding Model
# ==========================================

def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ==========================================
# Create Vector Store
# ==========================================

def create_vectorstore(chunks):

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


# ==========================================
# Save Vector Store
# ==========================================

def save_vectorstore(vectorstore):

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(VECTOR_DB_DIR)
    )


# ==========================================
# Complete Pipeline
# ==========================================

def process_pdf(pdf_path):

    print("Loading PDF...")

    documents = load_pdf(pdf_path)

    print(f"Pages Loaded : {len(documents)}")

    print("Splitting Document...")

    chunks = split_documents(documents)

    print(f"Chunks Created : {len(chunks)}")

    print("Creating Vector Database...")

    vectorstore = create_vectorstore(chunks)

    save_vectorstore(vectorstore)

    print("Vector Database Saved.")

    return len(chunks)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    pdf_path = Path("data/researchPaperIIIT.pdf")

    total_chunks = process_pdf(pdf_path)

    print()

    print("PDF Processed Successfully")

    print(f"Total Chunks : {total_chunks}")