import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from tqdm import tqdm
import time

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

def chat():
    print("Welcome to the Old English ChatBot! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Construct the prompt to force Old English replies
        old_english_prompt = """Respond in Shakespearen English
        context:{context}
        Question:{question}
        """

        # Create the prompt template
        chat_prompt = ChatPromptTemplate.from_template(old_english_prompt)

        # Initialize the LLMChain
        llm_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | chat_prompt
            | llm
            | StrOutputParser()
        )

         # Invoke the LLMChain and stream the result
        response = llm_chain.invoke(user_input)
        print("OE-Bot:", response)

# Start the Chat
chat()