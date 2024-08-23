import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser, Document
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pprint import pprint
from operator import itemgetter

# Update the import statement for HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Retrieve environment variables
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "False") == "True"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag system chr")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")  # Replace with your actual variable name if different

# Check if the query was provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python query_data.py \"<query>\"")
    sys.exit(1)

query = sys.argv[1]

# Define helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Example function to load documents (replace with your actual document loading code)
def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # Assuming the Document class is used to create documents
                doc = Document(page_content=content, metadata={"source": filename})
                documents.append(doc)
    return documents

# Load your documents (ensure this directory path is correct for your use case)
directory_path = 'huberman-lab-transcripts'

documents = load_documents(directory_path)

# Initialize the text splitter and split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(documents)

# Initialize the vector store and embeddings
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# Initialize vector store with documents
vectorstore = Chroma.from_documents(documents=all_splits, embedding=bge_embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Load prompt from HuggingFace hub or similar
prompt = hub.pull("rlm/rag-prompt", api_key=LANGCHAIN_API_KEY)

# Initialize language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=CHATGPT_API_KEY)

# Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Execute the chain with the provided query
print("Running RAG chain with query:", query)
result = rag_chain.invoke(query)
print("Result:")
pprint(result)

# Quote resources setup
rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

# Get results with source
result_with_source = rag_chain_with_source.invoke(query)
print("Result with source:")
pprint(result_with_source)
