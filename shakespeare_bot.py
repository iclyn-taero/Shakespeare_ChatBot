import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm
import time  # For simulating a long task (optional)

# Load the .env file
load_dotenv(dotenv_path="/Users/icemac/Documents/Projects/Shakespeare_ChatBot/.env")

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "ShakespeareChatBot/1.0"

# Initialize the LLM instance
llm = ChatOllama(model="llama3")

# URLs to Shakespearean text
urls = [
    "https://www.gutenberg.org/cache/epub/1513/pg1513-images.html",
    "https://www.gutenberg.org/files/1533/1533-h/1533-h.htm",
    "https://www.gutenberg.org/files/1514/1514-h/1514-h.htm"
]

# Create documents for the LLM from the URLs with loading spinner
with tqdm(total=len(urls), desc="Loading...", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed} [ {rate_fmt} ]") as pbar:
    docs = []
    for url in urls:
        docs.append(WebBaseLoader(url).load())
        pbar.update(1)
        time.sleep(1)  # Simulate a delay for each URL loading

docs_list = [item for sublist in docs for item in sublist]

# Break the data into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# Convert document to embeddings and store them
embeddings = OllamaEmbeddings(model='nomic-embed-text')  # Updated embedding instance
vector_store = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings  # Use embedding parameter instead of embedding_function
)

retriever = vector_store.as_retriever()

# Test the pipeline before using RAG
before_rag_template = "What is {topic}"
print("The response before using RAG:\n")
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | llm | StrOutputParser()

print(before_rag_chain.invoke({"topic": "Romeo"}))

# After rag using rag. Forcing it to use the shakespearen text to learn from and have context of
print("\n The reponse after using RAG")
after_rag_template = """Answer the question based only on the following 
context:{context}
Question:{question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | llm
    | StrOutputParser()
)
print(after_rag_chain.invoke("How did Romeo Fall?"))