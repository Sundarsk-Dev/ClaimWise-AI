import os
import shutil
import json
from typing import List, Tuple, Optional
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re # For extracting clause numbers/text

# Import chromadb for explicit persistent client
import chromadb

# --- Configuration ---
VECTOR_DB_PATH = "./insurance_chroma_db"
# IMPORTANT: Now using your custom model with the baked-in system prompt
LLM_MODEL_NAME = "insurance-analyst" 
MAX_RETRIEVED_CHUNKS = 3 # Keeping it at 3 for better context

# --- Global Variables for RAG components ---
vectorstore: Optional[Chroma] = None
llm: Optional[Ollama] = None

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
                print(f"Skipping unsupported file type: {file_path}")
                continue
            
            print(f"Loading {file_path}...")
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
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
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def get_embeddings_model():
    """Returns the SentenceTransformer embeddings model."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks):
    """
    Creates and persists a ChromaDB vector store from document chunks.
    This function is intended to be called when a new store needs to be created.
    It will clear any existing directory at VECTOR_DB_PATH before creation.
    """
    global vectorstore
    embedding_model = get_embeddings_model()
    collection_name = "insurance_policy_collection"

    # Ensure directory is clean before creating new
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            print(f"Cleared existing vector store directory at {VECTOR_DB_PATH} for new creation.")
        except Exception as e:
            print(f"Could not clear existing vector store directory at {VECTOR_DB_PATH}: {e}")
    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    persistent_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    print("Creating new vector store (this might take a moment)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=persistent_client, 
        collection_name=collection_name
    )
    
    print(f"Vector store created and persisted to {VECTOR_DB_PATH} with {len(chunks)} chunks.")
    return vectorstore

def setup_llm():
    """Initializes the Ollama LLM model."""
    global llm
    if llm is None:
        try:
            # Removed request_timeout as it's not supported by your LangChain version
            llm = Ollama(model=LLM_MODEL_NAME, temperature=0.1) 
            
            print(f"Attempting to test LLM: {LLM_MODEL_NAME}. This might take a while if the model is loading...")
            test_response = llm.invoke("Hello, are you working?")
            print(f"LLM test response: {test_response[:50]}...")
            print(f"Initialized LLM: {LLM_MODEL_NAME}")
        except Exception as e:
            print(f"Error initializing Ollama LLM with model '{LLM_MODEL_NAME}': {e}")
            print("Please ensure Ollama server is running and the model is downloaded correctly.")
            print("If the model is large and your system has limited RAM/GPU, it might take a long time to load or fail.")
            llm = None # Set to None if initialization fails
    return llm

# --- RAG Query and Decision Logic ---

def parse_query_and_decide(input_query: str) -> DecisionOutput:
    """
    Parses the input query, retrieves relevant clauses, and makes a decision
    using the LLM, outputting a structured JSON response.
    """
    global vectorstore, llm

    if vectorstore is None or llm is None:
        raise ValueError("Documents not processed or LLM not initialized. Please upload files first.")

    # 1. Retrieve relevant clauses based on the input query
    print(f"\n--- Retrieving relevant clauses for query: '{input_query}' ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVED_CHUNKS})
    relevant_docs = retriever.invoke(input_query)

    if not relevant_docs:
        print("No relevant clauses found in the documents.")
        return DecisionOutput(
            Decision="Insufficient Information",
            Justification="No relevant information or clauses could be retrieved from the provided documents based on your query.",
            ClausesUsed=[]
        )

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print("--- Retrieved Context (Top Chunks) ---")
    for i, doc in enumerate(relevant_docs):
        print(f"Chunk {i+1} (Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Page: {doc.metadata.get('page', 'N/A') + 1}):")
        print(f"  Content: {doc.page_content[:200]}...\n")

    # 2. Construct the specialized prompt for structured parsing and decision
    parser = PydanticOutputParser(pydantic_object=DecisionOutput)

    # Simplified Prompt Template - System prompt is now in Modelfile
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
    
    print("\n--- Sending Prompt to LLM ---")
    # print(chain_input.text[:1000] + "...") # Uncomment for debugging the full prompt
    
    # 3. Get response from LLM
    try:
        raw_llm_response = llm.invoke(chain_input.to_string())
        # print("\n--- Raw LLM Response ---") # Uncomment for debugging the raw LLM output
        # print(raw_llm_response)

        # 4. Parse the LLM's response into the Pydantic model
        try:
            parsed_output = parser.parse(raw_llm_response)
            print("\n--- Successfully Parsed LLM Output ---")
            return parsed_output
        except Exception as parse_error:
            print(f"Warning: LLM did not return perfect JSON. Attempting manual extraction. Error: {parse_error}")
            
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
        print(f"Error invoking LLM: {llm_error}")
        return DecisionOutput(
            Decision="Error",
            Justification=f"An error occurred while communicating with the LLM: {llm_error}",
            ClausesUsed=[]
        )

# --- Gradio UI Functions ---

def process_files(files):
    """
    Handles file uploads, processes them into the vector store,
    and initializes the LLM.
    """
    global vectorstore, llm

    if not files:
        return "Please upload at least one document (PDF, DOCX, TXT).", None

    file_paths = [file.name for file in files]
    print(f"Received files for processing: {file_paths}")

    try:
        # Check if vector store exists and is populated
        vector_store_populated = False
        if os.path.exists(VECTOR_DB_PATH):
            try:
                temp_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
                temp_collection = temp_client.get_collection(name="insurance_policy_collection")
                if temp_collection.count() > 0:
                    vector_store_populated = True
            except Exception as e:
                print(f"ChromaDB check failed (likely collection not found or corrupted): {e}")
                pass 

        if vector_store_populated:
            print(f"Loading existing vector store from {VECTOR_DB_PATH}.")
            persistent_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            vectorstore = Chroma(
                client=persistent_client,
                collection_name="insurance_policy_collection",
                embedding_function=get_embeddings_model()
            )
            print(f"Loaded {vectorstore._collection.count()} chunks from existing vector store.")
        else:
            print("Vector store not found or not populated. Creating a new one...")
            documents = load_documents(file_paths)
            if not documents:
                return "No valid documents were loaded from the uploaded files. Ensure they are PDF, DOCX, or TXT.", None
            chunks = split_documents(documents)
            vectorstore = create_vector_store(chunks)

        # LLM is initialized at startup, no need to call setup_llm here.
        if vectorstore and llm: # Check if llm was successfully initialized at startup
            return "Documents processed and RAG model ready! You can now ask claim-related queries.", None
        else:
            return "Documents processed, but LLM initialization failed. Please check your Ollama server and try restarting the script. If the problem persists, consider using a smaller LLM model like 'phi3:mini' or ensuring GPU acceleration is active for Ollama."

    except Exception as e:
        print(f"Error during file processing: {e}")
        return f"An error occurred: {e}. Please try again.", None

def reset_state():
    """Resets global variables for vectorstore and LLM."""
    global vectorstore, llm
    vectorstore = None
    # Note: We don't reset llm to None here, as it's initialized once at startup.
    # If it failed, it stays None, and the user needs to restart the script.
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            print(f"Cleared persistent vector store at {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"Error clearing persistent vector store: {e}")
    
    return (
        gr.update(value=""),  # msg (textbox)
        gr.update(value=[]),  # chatbot_display (chatbot)
        gr.update(value=None), # file_output (file component)
        gr.update(value="State Cleared. Please re-upload documents.") # status_message (textbox)
    )

def chatbot_response(message, history):
    """
    Generates a structured JSON response to the user's claim query.
    """
    if history is None:
        history = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Thinking..."})
    yield "", history

    if vectorstore is None or llm is None:
        history[-1]["content"] = "Please upload documents and wait for processing before asking questions. If LLM initialization failed, restart the script."
        yield "", history
        return

    try:
        decision_output = parse_query_and_decide(message)
        
        response_str = f"**Decision:** {decision_output.Decision}\n"
        response_str += f"**Justification:** {decision_output.Justification}\n"
        response_str += f"**Clauses Used:**\n"
        if decision_output.ClausesUsed:
            for clause in decision_output.ClausesUsed:
                response_str += f"- {clause}\n"
        else:
            response_str += "- No specific clauses identified in decision explanation."

        history[-1]["content"] = response_str
        yield "", history

    except ValueError as ve:
        history[-1]["content"] = f"Error: {str(ve)}"
        yield "", history
    except Exception as e:
        print(f"Error during claim processing: {e}")
        history[-1]["content"] = "An internal error occurred while processing your claim. Please check server logs."
        yield "", history

# --- Initialize LLM at script startup ---
# This will attempt to load the LLM model as soon as the Python script runs.
print("Initializing Ollama LLM at startup...")
llm = setup_llm()
print("Ollama LLM initialization complete (or attempted).")

# --- Gradio Interface ---

with gr.Blocks(title="ClaimWise-AI with Rag", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 **ClaimWise-AI: Your Insurance Policy Analyst**
        Welcome to ClaimWise-AI! Upload your insurance policy documents (PDF, DOCX, TXT).
        I will then analyze your claim queries and provide a structured decision (Approved/Rejected/etc.)
        with clear justification, *strictly based on the provided documents*.
        
        **💡 Example Query:** "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        
        **Performance Note:** If the bot is slow, ensure your Ollama server is running and consider if your system has a compatible GPU. Running LLMs on CPU can be time-consuming.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### 📄 **Upload Policy Documents**
                Drag & drop your policy files here or click to browse.
                The more relevant documents you provide, the better I can assist!
                """
            )
            file_output = gr.File(
                label="Supported formats: PDF, DOCX, TXT", 
                file_count="multiple", 
                file_types=[".pdf", ".docx", ".txt"],
                elem_id="file-upload-box"
            )
            process_button = gr.Button("🚀 Process Documents and Initialize Bot", variant="primary")
            status_message = gr.Textbox(label="Processing Status", interactive=False, placeholder="Awaiting documents...")
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 **Claim Analysis Chat**")
            chatbot_display = gr.Chatbot(height=400, label="Claim Analysis Output", type='messages') 
            msg = gr.Textbox(label="Enter Your Claim Query", placeholder="e.g., 46-year-old male, knee surgery in Pune, 3-month-old insurance policy")
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Claim", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot_display, file_output, status_message], value="🧹 Clear All")

    process_button.click(
        process_files,
        inputs=[file_output],
        outputs=[status_message, gr.State(None)] 
    )

    submit_btn.click(
        chatbot_response,
        inputs=[msg, chatbot_display],
        outputs=[msg, chatbot_display],
    ).then(
        lambda: gr.update(value=""), 
        inputs=None,
        outputs=msg,
    )
    
    msg.submit(
        chatbot_response,
        inputs=[msg, chatbot_display],
        outputs=[msg, chatbot_display],
    ).then(
        lambda: gr.update(value=""),
        inputs=None,
        outputs=msg,
    )

    clear_btn.click(
        reset_state,
        inputs=[],
        outputs=[msg, chatbot_display, file_output, status_message] 
    )

    demo.launch(share=True, max_threads=10)
