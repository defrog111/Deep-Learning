"""
Minimal local RAG example with LangChain-style building blocks.

This example does not require an API key. It uses:
  - LangChain Document objects
  - LangChain text splitter
  - LangChain prompt/runnable pipeline
  - scikit-learn TF-IDF vectors for local retrieval

Install:
    pip install -r requirements.txt

Build the retrieval index:
    python rag_langchain_example.py build-index

Ask a question:
    python rag_langchain_example.py ask --question "What is LoRA useful for?"

The answer generator is intentionally simple and local: it selects relevant
sentences from retrieved context. If you later want a real LLM, replace
answer_from_context() with an OpenAI, Ollama, or Hugging Face chat model call.
"""

import argparse
import pickle
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = Path("knowledge_base")
INDEX_PATH = Path("rag_index.pkl")


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.txt")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path)},
            )
        )
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(args) -> None:
    documents = load_documents(Path(args.docs_dir))
    chunks = split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    with Path(args.index_path).open("wb") as file:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "chunks": chunks,
            },
            file,
        )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")
    print(f"Saved index to: {args.index_path}")


def load_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run `python rag_langchain_example.py build-index` first."
        )
    with index_path.open("rb") as file:
        return pickle.load(file)


def retrieve(question: str, index, top_k: int) -> list[Document]:
    query_vector = index["vectorizer"].transform([question])
    scores = cosine_similarity(query_vector, index["matrix"]).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results: list[Document] = []
    for rank, chunk_index in enumerate(ranked_indices, start=1):
        chunk = index["chunks"][chunk_index]
        chunk.metadata = {
            **chunk.metadata,
            "score": float(scores[chunk_index]),
            "rank": rank,
        }
        results.append(chunk)
    return results


def format_context(documents: list[Document]) -> str:
    context_parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        rank = doc.metadata.get("rank", "?")
        context_parts.append(f"[{rank}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(context_parts)


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def answer_from_context(inputs: dict) -> str:
    question = inputs["question"]
    context = "\n".join(
        line for line in inputs["context"].splitlines() if not line.startswith("[")
    )

    question_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question)
        if len(word) > 2
    }
    scored_sentences = []
    for sentence in split_sentences(context):
        sentence_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", sentence)
        }
        score = len(question_words & sentence_words)
        if score > 0:
            scored_sentences.append((score, sentence))

    if not scored_sentences:
        return "I could not find enough information in the retrieved documents."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    answer_sentences = [sentence for _, sentence in scored_sentences[:3]]
    return " ".join(answer_sentences)


def ask(args) -> None:
    index = load_index(Path(args.index_path))

    retriever = RunnableLambda(
        lambda question: retrieve(question, index=index, top_k=args.top_k)
    )
    prompt = PromptTemplate.from_template(
        "Answer the question using only the retrieved context.\n\n"
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Answer:"
    )

    chain = (
        {
            "question": RunnablePassthrough(),
            "documents": retriever,
        }
        | RunnableLambda(
            lambda values: {
                "question": values["question"],
                "context": format_context(values["documents"]),
                "documents": values["documents"],
            }
        )
        | RunnableLambda(
            lambda values: {
                **values,
                "prompt": prompt.format(
                    question=values["question"],
                    context=values["context"],
                ),
            }
        )
        | RunnableLambda(
            lambda values: {
                "answer": answer_from_context(values),
                "documents": values["documents"],
                "prompt": values["prompt"],
            }
        )
    )

    result = chain.invoke(args.question)

    print("\nAnswer:")
    print(StrOutputParser().invoke(result["answer"]))

    print("\nRetrieved sources:")
    for doc in result["documents"]:
        print(
            f"- rank={doc.metadata['rank']} "
            f"score={doc.metadata['score']:.4f} "
            f"source={doc.metadata['source']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Local RAG example.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--docs_dir", default=str(DOCS_DIR))
    build_parser.add_argument("--index_path", default=str(INDEX_PATH))
    build_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top_k", type=int, default=3)
    ask_parser.add_argument("--index_path", default=str(INDEX_PATH))
    ask_parser.set_defaults(func=ask)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
