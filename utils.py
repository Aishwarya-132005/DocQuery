import shutil

from config import VECTOR_DB_DIR

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def get_embedding_model():

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


def vector_db_exists():

    return VECTOR_DB_DIR.exists()


def load_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        folder_path=str(VECTOR_DB_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def delete_vectorstore():

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)


def get_total_documents():

    if not vector_db_exists():
        return 0

    vectorstore = load_vectorstore()

    return vectorstore.index.ntotal


def get_database_info():

    if not vector_db_exists():

        return {
            "status": "Database not found",
            "documents": 0,
        }

    vectorstore = load_vectorstore()

    return {
        "status": "Database Loaded",
        "documents": vectorstore.index.ntotal,
        "embedding_model": EMBEDDING_MODEL,
    }


if __name__ == "__main__":

    print(get_database_info())