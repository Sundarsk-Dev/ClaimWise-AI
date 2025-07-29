import os
import shutil
import json
from typing import List, Tuple, Optional
import streamlit as st 

# Updated imports for Ollama and SentenceTransformerEmbeddings to address deprecation warnings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re 

# Import chromadb for explicit persistent client
import chromadb

# --- Configuration ---
VECTOR_DB_PATH = "./insurance_chroma_db"
# Using the custom model created with the optimized Modelfile
LLM_MODEL_NAME = "insurance-analyst" 
MAX_RETRIEVED_CHUNKS = 3 # Number of top chunks to pass to the LLM

# --- Pydantic Model for Structured Output ---
class DecisionOutput(BaseModel):
    """
    Represents the structured output for an insurance claim decision.
    Amount field removed as per user request.
    """
    Decision: str = Field(description="The decision regarding the claim (e.g., 'Approved', 'Rejected', 'Under Review', 'Insufficient Information').")
    Justification: str = Field(description="A detailed explanation of the decision, referencing specific clauses from the context.")
    ClausesUsed: List[str] = Field(description="A list of specific clause IDs or direct quotes from clauses that support the decision.")

# --- Document Loading and Indexing Functions ---

def load_documents(file_paths: List[str]):
    """Loads documents from provided file paths based on their extension."""
    all_documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt" or ext == ".eml":
                loader = TextLoader(file_path)
            else:
                st.warning(f"Skipping unsupported file type: {file_path}")
                continue
            
            st.info(f"Loading {file_path}...")
            all_documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            
    return all_documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Adjusted: Slightly smaller chunk size
        chunk_overlap=100,    # Adjusted: Reduced overlap for more distinct chunks
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def get_embeddings_model():
    """Returns the SentenceTransformer embeddings model.
    Updated to use HuggingFaceEmbeddings as per LangChain's recommendation."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks):
    """
    Creates and persists a ChromaDB vector store from document chunks.
    This function is intended to be called when a new store needs to be created.
    It will clear any existing directory at VECTOR_DB_PATH before creation.
    """
    embedding_model = get_embeddings_model()
    collection_name = "insurance_policy_collection"

    # Ensure directory is clean before creating new
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            st.info(f"Cleared existing vector store directory at {VECTOR_DB_PATH} for new creation.")
        except Exception as e:
            st.error(f"Could not clear existing vector store directory at {VECTOR_DB_PATH}: {e}")
    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    persistent_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    with st.spinner("Creating new vector store (this might take a moment)..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            client=persistent_client, 
            collection_name=collection_name
        )
    
    st.success(f"Vector store created and persisted to {VECTOR_DB_PATH} with {len(chunks)} chunks.")
    return vectorstore

@st.cache_resource # Cache the LLM to avoid re-initializing on every rerun
def setup_llm():
    """Initializes the Ollama LLM model."""
    try:
        with st.spinner(f"Attempting to initialize LLM: {LLM_MODEL_NAME}. This might take a while if the model is loading..."):
            llm_instance = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.1) 
            
            # A quick test call to ensure the LLM is responsive
            test_response = llm_instance.invoke("Hello, are you working?")
            st.success(f"LLM test response: {test_response[:50]}...")
            st.success(f"Initialized LLM: {LLM_MODEL_NAME}")
            return llm_instance
    except Exception as e:
        st.error(f"Error initializing Ollama LLM with model '{LLM_MODEL_NAME}': {e}")
        st.error("Please ensure Ollama server is running and the model is downloaded correctly.")
        st.error("If the model is large and your system has limited RAM/GPU, it might take a long time to load or fail.")
        return None

# --- RAG Query and Decision Logic ---

def parse_query_and_decide(input_query: str) -> DecisionOutput:
    """
    Parses the input query, retrieves relevant clauses, and makes a decision
    using the LLM, outputting a structured JSON response.
    """
    # Access cached components from session state
    vectorstore = st.session_state.vectorstore
    llm = st.session_state.llm

    if vectorstore is None or llm is None:
        raise ValueError("RAG components (Vector Store or LLM) are not initialized. Please upload files first.")
    
    # 1. Retrieve relevant clauses based on the input query (no re-ranking)
    st.info(f"Retrieving {MAX_RETRIEVED_CHUNKS} relevant chunks for query: '{input_query}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVED_CHUNKS})
    relevant_docs = retriever.invoke(input_query) 

    if not relevant_docs:
        st.warning("No relevant clauses found in the documents.")
        return DecisionOutput(
            Decision="Insufficient Information",
            Justification="No relevant information or clauses could be retrieved from the provided documents based on your query.",
            ClausesUsed=[]
        )

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    st.info("Retrieved Context (Top Chunks):")
    for i, doc in enumerate(relevant_docs):
        st.info(f"Chunk {i+1} (Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Page: {doc.metadata.get('page', 'N/A') + 1}): {doc.page_content[:100]}...")


    # 3. Construct the prompt for the LLM. The system prompt is now in the Modelfile.
    parser = PydanticOutputParser(pydantic_object=DecisionOutput)

    # Simplified Prompt Template - System prompt is now in Modelfile
    # This template will be wrapped by the Modelfile's TEMPLATE instruction.
    prompt_template = PromptTemplate(
        template="""Provided Insurance Clauses:
        {context}
        ---
        
        Claim Query: "{query}"

        {format_instructions}
        """,
        input_variables=["context", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Combine context and query for the LLM
    chain_input = prompt_template.format_prompt(context=context_text, query=input_query)
    
    st.info("Sending Prompt to LLM for RAG analysis...")
    
    # 4. Get response from LLM
    try:
        with st.spinner("Getting decision from LLM..."):
            raw_llm_response = llm.invoke(chain_input.to_string())
        
        # 5. Parse the LLM's response into the Pydantic model
        try:
            parsed_output = parser.parse(raw_llm_response)
            st.success("Successfully Parsed LLM Output.")
            return parsed_output
        except Exception as parse_error:
            st.warning(f"LLM did not return perfect JSON. Attempting manual extraction. Error: {parse_error}")
            
            # Fallback: Try to extract structured parts manually if Pydantic parsing fails
            decision_match = re.search(r"\"Decision\":\s*\"([^\"]+)\"", raw_llm_response)
            justification_match = re.search(r"\"Justification\":\s*\"([^\"]+)\"", raw_llm_response)
            clauses_match = re.search(r"\"ClausesUsed\":\s*\[([^\]]*)\]", raw_llm_response, re.DOTALL)

            fallback_decision = decision_match.group(1) if decision_match else "Parsing Error"
            fallback_justification = justification_match.group(1) if justification_match else "Could not fully parse justification from LLM response."
            
            fallback_clauses = []
            if clauses_match:
                clause_text = clauses_match.group(1).strip()
                fallback_clauses = [c.strip().strip('"') for c in clause_text.split(',') if c.strip()]
            
            return DecisionOutput(
                Decision=fallback_decision,
                Justification=fallback_justification,
                ClausesUsed=fallback_clauses
            )

    except Exception as llm_error:
        st.error(f"Error invoking LLM: {llm_error}")
        return DecisionOutput(
            Decision="Error",
            Justification=f"An error occurred while communicating with the LLM: {llm_error}",
            ClausesUsed=[]
        )

def get_general_chat_response(input_query: str) -> str:
    """
    Gets a general conversational response from the LLM without RAG.
    """
    llm = st.session_state.llm
    if llm is None:
        return "Chatbot not initialized. Please restart the application."
    
    st.info("Sending Prompt to LLM for general chat...")
    try:
        with st.spinner("Thinking..."):
            # A simple prompt for general conversation. The Modelfile's SYSTEM prompt
            # already instructs the model to act as an analyst for insurance queries
            # and naturally for general ones. So, we just pass the raw query here.
            response = llm.invoke(input_query)
            return response
    except Exception as e:
        st.error(f"Error getting general chat response: {e}")
        return "I'm sorry, I encountered an error while trying to respond."


# --- Streamlit UI Layout ---

st.set_page_config(page_title="ClaimWise AI", layout="wide")

st.title("🏥 ClaimWise AI: Your Insurance Policy Analyst")
st.markdown(
    """
    Welcome to ClaimWise AI! Upload your insurance policy documents (PDF, DOCX, TXT).
    I will then analyze your claim queries and provide a structured decision (Approved/Rejected/etc.)
    with clear justification, *strictly based on the provided documents*.
    
    **💡 Example Query:** "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    
    **Performance Note:** Initial model loading and document processing times are dependent on your system's CPU and RAM. Subsequent query responses should be faster.
    """
)

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state: # New session state for mode
    st.session_state.mode = "Insurance Analyst" # Default mode

# Initialize LLM once at startup and cache it
# Ollama is explicitly initialized and tested here
if st.session_state.llm is None:
    st.session_state.llm = setup_llm()

# Sidebar for file upload and status
with st.sidebar:
    st.header("📄 Upload Policy Documents")
    st.write("Drag & drop your policy files here or click to browse.")
    st.write("The more relevant documents you provide, the better I can assist!")
    
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )

    process_button = st.button("🚀 Process Documents and Initialize RAG")

    if process_button and uploaded_files:
        file_paths = []
        # Create a temporary directory for uploaded files
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_file_path)

        with st.spinner("Processing documents..."):
            documents = load_documents(file_paths)
            if documents:
                chunks = split_documents(documents)
                st.session_state.vectorstore = create_vector_store(chunks)
                st.success("Documents processed and RAG model ready! You can now ask claim-related queries.")
            else:
                st.error("No valid documents were loaded from the uploaded files. Ensure they are PDF, DOCX, or TXT.")
        
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            st.info("Cleaned up temporary uploaded files.")

    elif process_button and not uploaded_files:
        st.warning("Please upload at least one document (PDF, DOCX, TXT).")

    st.markdown("---")
    # Mode Toggle
    st.header("🤖 Chat Mode")
    st.session_state.mode = st.radio(
        "Select Chat Mode:",
        ("Insurance Analyst", "General Chatbot"),
        index=0 if st.session_state.mode == "Insurance Analyst" else 1, # Set initial state
        key="chat_mode_radio"
    )
    st.info(f"Current mode: **{st.session_state.mode}**")


    st.markdown("---")
    if st.button("🧹 Clear All State and Documents"):
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
            st.info(f"Cleared persistent vector store at {VECTOR_DB_PATH}")
        st.session_state.vectorstore = None
        st.session_state.messages = []
        # Re-initialize LLM as it's a cached resource
        st.session_state.llm = setup_llm() 
        st.success("Application state cleared. Please re-upload documents.")
        st.experimental_rerun() # Rerun to clear chat display

# Main chat interface
st.header("💬 Claim Analysis Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine response based on mode
    with st.chat_message("assistant"):
        if st.session_state.mode == "Insurance Analyst":
            # In Insurance Analyst mode, always use RAG
            with st.spinner("Analyzing claim..."):
                try:
                    decision_output = parse_query_and_decide(prompt)
                    response_str = f"**Decision:** {decision_output.Decision}\n"
                    response_str += f"**Justification:** {decision_output.Justification}\n"
                    response_str += f"**Clauses Used:**\n"
                    if decision_output.ClausesUsed:
                        for clause in decision_output.ClausesUsed:
                            response_str += f"- {clause}\n"
                    else:
                        response_str += "- No specific clauses identified in decision explanation."
                    
                    st.markdown(response_str)
                    st.session_state.messages.append({"role": "assistant", "content": response_str})

                except ValueError as ve:
                    error_msg = f"Error: {str(ve)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An internal error occurred while processing your claim: {e}. Please check server logs."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        elif st.session_state.mode == "General Chatbot":
            # Simple keyword check for intent routing even in general mode
            insurance_keywords = ["policy", "claim", "coverage", "deductible", "premium", 
                                  "hospital", "surgery", "accident", "benefit", "exclusion", 
                                  "waiting period", "medical", "insurance", "46m", "m", "male",
                                  "female", "age", "year old", "month policy", "knee surgery", 
                                  "pune", "mumbai", "procedure", "diagnosis", "treatment"]
            
            # Check if any insurance-related keyword is in the prompt AND documents have been processed
            if st.session_state.vectorstore is not None and any(keyword in prompt.lower() for keyword in insurance_keywords):
                st.warning("Detected insurance-related query in General Chatbot mode. Switching to Insurance Analyst logic.")
                with st.spinner("Analyzing claim (detected insurance intent)..."):
                    try:
                        decision_output = parse_query_and_decide(prompt)
                        response_str = f"**Decision:** {decision_output.Decision}\n"
                        response_str += f"**Justification:** {decision_output.Justification}\n"
                        response_str += f"**Clauses Used:**\n"
                        if decision_output.ClausesUsed:
                            for clause in decision_output.ClausesUsed:
                                response_str += f"- {clause}\n"
                        else:
                            response_str += "- No specific clauses identified in decision explanation."
                        
                        st.markdown(response_str)
                        st.session_state.messages.append({"role": "assistant", "content": response_str})
                    except ValueError as ve:
                        error_msg = f"Error: {str(ve)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"An internal error occurred while processing your claim: {e}. Please check server logs."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                # General conversation
                general_response = get_general_chat_response(prompt)
                st.markdown(general_response)
                st.session_state.messages.append({"role": "assistant", "content": general_response})

