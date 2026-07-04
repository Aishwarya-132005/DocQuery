from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from rag import ask_question, load_llm, load_embeddings


questions = [
    "What is Retrieval Augmented Generation?",
    "What is FAISS?",
    "What embedding model is used?",
    "How is the PDF split?",
    "What is LangChain?"
]

ground_truths = [
    "Retrieval Augmented Generation combines retrieval with text generation.",
    "FAISS is a vector similarity search library.",
    "The embedding model is sentence-transformers/all-MiniLM-L6-v2.",
    "The PDF is split using RecursiveCharacterTextSplitter.",
    "LangChain is a framework for building LLM applications."
]


def build_dataset():

    answers = []

    contexts = []
 
    for question in questions:

        response = ask_question(question)

        answers.append(
            response["answer"]
        )

        contexts.append(
            [
                document.page_content
                for document in response["context"]
            ]
        )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    return dataset


def evaluate_rag():

    dataset = build_dataset()

    # Use the project's HuggingFace LLM and embeddings for RAGAS evaluation
    llm = load_llm()
    embeddings = load_embeddings()

    results = evaluate(

        dataset,

        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],

        llm=llm,
        embeddings=embeddings,

    )

    return results


if __name__ == "__main__":

    results = evaluate_rag()

    print(results)

    dataframe = results.to_pandas()

    print(dataframe)

    dataframe.to_csv(
        "ragas_results.csv",
        index=False,
    )

    print("Results saved to ragas_results.csv")