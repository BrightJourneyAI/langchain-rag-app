import streamlit as st
from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.retrievers import RetrieverOutputLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage

# Init Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR = BASE_DIR / 'db'
DB_DIR.mkdir(parents=True, exist_ok=True)

def persist_file(upload):
    """
    persist_file Will persist the provided file to the file system

    Args:
        upload (UploadedFile): The file uploaded via the UI
    """
    save_path = Path(DATA_DIR, upload.name)
    with open(save_path, mode="wb") as w:
        w.write(upload.getvalue())

    if save_path.exists():
        st.success(f"File {upload.name} is successfully saved")

def init_ui():
    """
    init_ui Initializes the UI
    """
    st.set_page_config(page_title="Langchain RAG Bot", layout="wide")
    st.title("Langchain RAG Bot")

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm here to help. Ask me anything!")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("Document Capture")
        st.write("Please select a single document to use as context")
        st.markdown("**Please fill the below form :**")
        with st.form(key="Form", clear_on_submit = True):
            uploaded_file = st.file_uploader("Upload", type=["pdf"], key="pdf_upload")
            submit = st.form_submit_button(label="Upload")

        if submit:
            persist_file(uploaded_file)
            vector_store = init_vector_store()
            st.session_state.vector_store = vector_store
    if st.session_state.vector_store is not None:
        init_chat_interface()


def init_vector_store():
    """
    Initializes and returns ChromaDB vector store from document chunks

    Returns:
        ChromaDB: Initialized vector store
    """
    # Get the first file - in reality this would be more robust
    files = [f for f in DATA_DIR.iterdir() if f.is_file]
    if not files:
        st.error("No files uploaded")
        return None

    # Get the path to the first file in the directory
    first_file = files[0].resolve()
    # Use the PDF loader in Langchain to fetch the document text
    loader = PyPDFLoader(first_file)
    document = loader.load_and_split()

    # Now we initialise the text splitter we will use on the document
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Lastly, we initialise the vector store using the split document
    vector_store = Chroma.from_documents(
        documents=document_chunks,
        embedding=OllamaEmbeddings(),
        persist_directory=str(DB_DIR),
        collection_name="pdf_v_db" # Important if you want to reference the DB later
    )

    return vector_store

def get_related_context(vector_store: Chroma) -> RetrieverOutputLike:
    """
    Will retrieve the relevant context based on the user's query 
    using Approximate Nearest Neighbor search (ANN)

    Args:
        vector_store (Chroma): The initialized vector store with context

    Returns:
        RetrieverOutputLike: The chain component to be used with the LLM
    """

    # Specify the model to use
    llm = Ollama(model="llama3")

    # Here we are using the vector store as the source
    retriever = vector_store.as_retriever()

    # Create a prompt that will be used to query the vector store for related content
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    # Create the chain element which will fetch the relevant content from ChromaDB
    chain_element = create_history_aware_retriever(llm, retriever, prompt)
    return chain_element


def get_context_aware_prompt(context_chain: RetrieverOutputLike) -> Runnable:
    """
    Combined the chain element to fetch content with one that then creates the 
    prompt used to interact with the LLM

    Args:
        context_chain (RetrieverOutputLike): The retriever chain that can 
            fetch related content from ChromaDB

    Returns:
        Runnable: The full runnable chain that can be executed
    """

    # Specify the model to use
    llm = Ollama(model="llama3")

    # A standard prompt template which combined chat history with user query
    # NOTE: You must pass the context into the system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer the users questions. Use provided context to answer the question as accurately as possible:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # This method creates a chain for passing documents to a LLM
    docs_chain = create_stuff_documents_chain(llm, prompt)

    # Now we merge the context chain & docs chain to form the full prompt
    rag_chain = create_retrieval_chain(context_chain, docs_chain)
    return rag_chain

def get_response(user_query: str) -> str:
    """
    Will use the query to fetch context & form a query to send to an LLM.
    Responds with the result of the query 

    Args:
        user_query (str): Query input but user

    Returns:
        str: Answer from the LLM
    """
    context_chain = get_related_context(st.session_state.vector_store)
    rag_chain = get_context_aware_prompt(context_chain)

    res = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return res["answer"]


def init_chat_interface():
    """
    Initializes a chat interface which will leverage our rag chain & a local LLM
    to answer questions about the context provided
    """

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        
        # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        

if __name__ == "__main__":
    init_ui()
