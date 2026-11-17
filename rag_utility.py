# rag_utility.py
import os
import logging
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
# Optional fallback – only imported when needed
# from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------------------------------------------------
# Logging configuration – you will see these lines in Streamlit Cloud logs
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Global objects (created once per container start)
# ----------------------------------------------------------------------
# You can change the model name to any sentence‑transformer you like.
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------------------------------------------
# Helper: write the uploaded Streamlit file to a real temporary file
# ----------------------------------------------------------------------
def _write_to_temp(uploaded_file) -> str:
    """
    Returns the absolute path of a temporary file that contains the uploaded
    PDF bytes. The caller is responsible for deleting the file when done.
    """
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

# ----------------------------------------------------------------------
# Main: process a PDF and store its chunks in a persistent Chroma DB
# ----------------------------------------------------------------------
def process_document_to_chroma_db(uploaded_file) -> None:
    """
    1️⃣ Save the uploaded file to a temporary location.
    2️⃣ Load the PDF (PyPDFLoader → fallback to UnstructuredPDFLoader).
    3️⃣ Chunk the text.
    4️⃣ Persist the chunks in a Chroma collection.
    """
    # -------------------------------------------------
    # 1️⃣ Save to a real file on disk
    # -------------------------------------------------
    temp_path = _write_to_temp(uploaded_file)
    log.info(f"Saved uploaded PDF to temporary path: {temp_path}")

    # -------------------------------------------------
    # 2️⃣ Load the PDF
    # -------------------------------------------------
    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        log.info(f"Loaded {len(documents)} pages with PyPDFLoader")
    except Exception as e:
        log.warning(f"PyPDFLoader failed ({e}); trying UnstructuredPDFLoader")
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(temp_path)
            documents = loader.load()
            log.info(f"Loaded {len(documents)} pages with UnstructuredPDFLoader")
        except Exception as ue:
            # Clean up the temp file before bubbling up
            os.unlink(temp_path)
            log.exception("Both PDF loaders failed")
            raise RuntimeError(
                "Unable to parse the uploaded PDF. See logs for the exact error."
            ) from ue

    # -------------------------------------------------
    # 3️⃣ Chunk the text
    # -------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info(f"Split into {len(chunks)} text chunks")

    # -------------------------------------------------
    # 4️⃣ Persist to Chroma
    # -------------------------------------------------
    # Use a folder that lives next to this file – it survives across
    # Streamlit reruns as long as the container isn’t rebuilt.
    persist_dir = Path(__file__).parent / "doc_vectorstore"
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(persist_dir),
    )
    # Explicitly write the DB to disk (good practice)
    vectordb.persist()
    log.info(f"Vector store persisted at {persist_dir}")

    # -------------------------------------------------
    # 5️⃣ Clean up the temporary PDF file
    # -------------------------------------------------
    os.unlink(temp_path)
    log.info("Temporary PDF file removed")

# ----------------------------------------------------------------------
# Retrieval / QA
# ----------------------------------------------------------------------
def answer_question(user_question: str) -> str:
    """
    Loads the persisted Chroma DB, builds a retriever and runs a
    RetrievalQA chain with the Groq LLM.
    """
    # -----------------------------------------------------------------
    # Load the vector store (must point to the same folder used above)
    # -----------------------------------------------------------------
    persist_dir = Path(__file__).parent / "doc_vectorstore"
    if not persist_dir.is_dir():
        raise RuntimeError(
            f"Vector store directory '{persist_dir}' does not exist. "
            "Did you run the document‑ingestion step?"
        )

    vectordb = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # -----------------------------------------------------------------
    # LLM – Groq
    # -----------------------------------------------------------------
    # Make sure you have set GROQ_API_KEY in your .env or Streamlit secrets.
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="groq/gpt-4o-mini",   # <-- change to any Groq model you have access to
        temperature=0,
        # optional: you can pass `api_key=os.getenv("GROQ_API_KEY")` explicitly
    )

    # -----------------------------------------------------------------
    # RetrievalQA chain
    # -----------------------------------------------------------------
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # “stuff” works well for small‑to‑medium docs
        retriever=retriever,
        return_source_documents=False,
    )

    # -----------------------------------------------------------------
    # Run the query
    # -----------------------------------------------------------------
    response = qa_chain.invoke({"query": user_question})
    # The exact key depends on the chain version; most recent versions return
    # a dict with a single key "result".
    answer = response.get("result") or response.get("output_text") or str(response)
    return answer
