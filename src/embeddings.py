import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


def bge_small_en_v15_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def split_for_vectorstore(path: str, chunk_size=200, chunk_overlap=64):
    print("Create vectorstore " + path + "...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    raw_documents = TextLoader(path).load()
    documents = text_splitter.split_documents(raw_documents)
    return documents


def create_vectorstore(documents, embeddings):
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def save_vectorstore(name: str, vectorstore: FAISS):
    vectorstore.save_local(name)


def load_vectorstore(name: str, embeddings):
    if not os.path.exists(name):
        print(f"Embedding not found for {name}")
        return None

    vectorstore = FAISS.load_local(
        name,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
