import os
from pathlib import Path
from langchain_community.vectorstores import FAISS

from embeddings import (
    bge_small_en_v15_embeddings,
    create_vectorstore,
    load_vectorstore,
    save_vectorstore,
    split_for_vectorstore,
)
from split_text import split_on_markdown
from extract_data import extract_pdf_data


def get_documents_embeddings(input, k=5):
    embeddings = bge_small_en_v15_embeddings()
    documents_embedding_name = "embeddings/documents_embedding"
    documents_vectorstore = load_vectorstore(documents_embedding_name, embeddings)
    document_context = documents_vectorstore.similarity_search(input, k)
    document_context = " ".join([doc.page_content for doc in document_context])
    return document_context


def create_and_save_vectorstore(markdownpath):
    with open(markdownpath) as f:
        markdown = f.read()
    document_embedding_name = "embeddings/documents_embedding"
    embeddings = bge_small_en_v15_embeddings()
    documents = split_on_markdown(markdown)
    vectorstore = create_vectorstore(documents, embeddings)
    save_vectorstore(document_embedding_name, vectorstore)


def create_all_vectorstore(folder_path="../doc"):
    embeddings = bge_small_en_v15_embeddings()
    documents_embedding_name = "embeddings/documents_embedding"
    documents_vectorstore = load_vectorstore(documents_embedding_name, embeddings)

    for filename in os.listdir(folder_path):
        print("Processing file: " + str(filename))
        if os.path.splitext(filename)[1] in [".ttl"]:
            print("Processing ttl file ...")
            name = "embeddings/" + Path(filename).stem + "_embedding"
            path = os.path.join(folder_path, filename)
            vectorstore = load_vectorstore(name, embeddings)
            if vectorstore is None:
                print("Extracting data from ttl ...")
                documents = split_for_vectorstore(path)
                vectorstore = create_vectorstore(documents, embeddings)
                save_vectorstore(name, vectorstore)
        elif os.path.splitext(filename)[1] in [".pdf"]:
            path = os.path.join(folder_path, filename)
            markdownpath = (
                folder_path
                + "/"
                + Path(filename).stem
                + "/"
                + Path(filename).stem
                + ".md"
            )
            if not os.path.exists(markdownpath):
                print("Extracting data from pdf ...")
                extract_pdf_data(path)
            name = "embeddings/" + Path(filename).stem + "_embedding"
            path = os.path.join(folder_path, filename)
            vectorstore = load_vectorstore(name, embeddings)
            if vectorstore is None:
                documents = split_on_markdown(markdownpath)
                vectorstore = create_vectorstore(documents, embeddings)
                save_vectorstore(name, vectorstore)
            if documents_vectorstore is None:
                documents_vectorstore = FAISS.from_texts([""], embeddings)
            try:
                documents_vectorstore.merge_from(vectorstore)
                save_vectorstore(documents_embedding_name, documents_vectorstore)
            except ValueError:
                print("Did not merge vectorstores. Ids already exist.")
        elif os.path.splitext(filename)[1] in [".md"]:
            name = "embeddings/" + Path(filename).stem + "_embedding"
            path = os.path.join(folder_path, filename)
            vectorstore = load_vectorstore(name, embeddings)
            if vectorstore is None:
                documents = split_on_markdown(path)
                vectorstore = create_vectorstore(documents, embeddings)
                save_vectorstore(name, vectorstore)
            if documents_vectorstore is None:
                documents_vectorstore = FAISS.from_texts([""], embeddings)
            try:
                documents_vectorstore.merge_from(vectorstore)
                save_vectorstore(documents_embedding_name, documents_vectorstore)
            except ValueError as e:
                print("Did not merge vectorstores. Ids already exist.")
        elif os.path.splitext(filename)[1] in [".txt", ".json"]:
            name = "embeddings/" + Path(filename).stem + "_embedding"
            path = os.path.join(folder_path, filename)
            vectorstore = load_vectorstore(name, embeddings)
            if vectorstore is None:
                documents = split_for_vectorstore(path)
                vectorstore = create_vectorstore(documents, embeddings)
                save_vectorstore(name, vectorstore)
            if documents_vectorstore is None:
                documents_vectorstore = FAISS.from_texts([""], embeddings)
            try:
                documents_vectorstore.merge_from(vectorstore)
                save_vectorstore(documents_embedding_name, documents_vectorstore)
            except ValueError as e:
                print("Did not merge vectorstores. Ids already exist.")


def main():
    create_all_vectorstore()


if __name__ == "__main__":
    main()
