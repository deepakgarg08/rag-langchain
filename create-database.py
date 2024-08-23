import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Assuming 'Document' is a class that needs to be defined or imported:
from langchain.schema import Document

def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Extract URL and Title
    url = lines[0].strip()
    title = lines[2].strip()
    if (title == "\ufeff" or title == ""):
        title = lines[3].strip()

    # Extract page content after "TRANSCRIPT"
    transcript_index = lines.index('TRANSCRIPT\n')
    page_content = ''.join(lines[transcript_index + 1:])

    return Document(page_content=page_content, metadata={'source': url, 'title': title, 'file_path': file_path})


def create_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            doc = process_txt_file(os.path.join(directory_path, filename))
            documents.append(doc)
    return documents


def main():
    # Example usage
    directory_path = 'huberman-lab-transcripts'
    docs = create_documents_from_directory(directory_path)
    print("len(docs):", len(docs))
    print("docs[0].metadata ....:", docs[0].metadata)
    print("docs[0].page_content[:200] ....:", docs[0].page_content[:200])

    # 2 - Splitting the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print("len(all_splits):", len(all_splits))
    print("all_splits[1].page_content:", all_splits[1].page_content)

    # 3 - Embedding chunks and loading into a vector database
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}  # Set True to compute cosine similarity
    print("working on embedding..., please wait")
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        # model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=bge_embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(
        # "How do I find my temperature minimum?"
        "whats the ideal body temperature?"
    )
    print("len(retrieved_docs):", len(retrieved_docs))


if __name__ == "__main__":
    main()
