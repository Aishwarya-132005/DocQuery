from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains import (
    create_retrieval_chain,
)

from config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    JUDGE_LLM_MODEL,
    TOP_K,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


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
# Load Vector Store
# ==========================================

def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = FAISS.load_local(
        folder_path=str(VECTOR_DB_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# ==========================================
# Load Hugging Face Llama
# ==========================================

def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,      #LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
        task="conversational",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def load_judge_llm():

    llm = HuggingFaceEndpoint(
        repo_id=JUDGE_LLM_MODEL,
        task="conversational",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


# ==========================================
# Prompt
# ==========================================

#CONTEXT BASED TEMPLATE-PROMTE
PROMPT = """\
You are a precise and helpful AI assistant that answers questions based strictly on the provided context.

Instructions:
1. Read the context carefully and answer the question directly.
2. Use ONLY information from the context below. Do not add external knowledge.
3. Keep your answer concise, accurate, and to the point.
4. Reference specific details from the context to support your answer.
5. If the context does not contain enough information to answer, say exactly: "I could not find the answer in the PDF."

Context:
{context}

Question: {input}

Answer:"""

prompt = PromptTemplate(

    template=PROMPT,

    input_variables=[
        "context",
        "input",
    ],
)


# ==========================================
# Retrieval Chain
# ==========================================

def create_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(

        search_type="mmr",      #Maximum Marginal Relevance (MMR) retrieval.

        search_kwargs={
            "k": TOP_K,
            "fetch_k": TOP_K * 3,
            "lambda_mult": 0.7,
        }

    )

    llm = load_llm()

    document_chain = create_stuff_documents_chain(

        llm,

        prompt,

    )

    retrieval_chain = create_retrieval_chain(

        retriever,

        document_chain,

    )

    return retrieval_chain



# Ask Question


def ask_question(question):

    chain = create_chain()

    response = chain.invoke(

        {
            "input": question
        }

    )

    return response


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    question = input(
        "Ask Question : "
    )

    result = ask_question(
        question
    )

    print()

    print("Answer\n")

    print(result["answer"])

    print()

    print("Retrieved Documents\n")

    for document in result["context"]:

        print("=" * 80)

        print(document.metadata)

        print()

        print(document.page_content[:500])