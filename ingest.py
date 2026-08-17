"""
ingest.py

Loads every document from data/, splits them into overlapping chunks,
embeds each chunk, and builds a local FAISS vector index on disk.

Run this once, and again any time you add/change documents in data/:
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"


def load_documents():
    """Load every PDF and .txt file in data/ as LangChain Document objects.

    Each Document carries metadata automatically (source filename, page number
    for PDFs). We keep this metadata all the way through the pipeline so the
    chatbot can cite exactly which document/page an answer came from.
    """
    documents = []

    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    # 3GPP specs are commonly distributed as .docx (inside a zip on the 3GPP site),
    # so we support that format directly rather than requiring manual conversion to PDF.
    docx_loader = DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader)

    pdf_files = list(pdf_loader.load())
    txt_files = list(txt_loader.load())
    docx_files = list(docx_loader.load())

    documents.extend(pdf_files)
    documents.extend(txt_files)
    documents.extend(docx_files)

    print(f"  Found {len(pdf_files)} PDF page(s), {len(txt_files)} txt file(s), {len(docx_files)} docx file(s) in {DATA_DIR}/")

    if not documents:
        raise ValueError(
            f"No .pdf, .txt, or .docx files found in {DATA_DIR}/. "
            f"Check that your file actually landed inside the '{DATA_DIR}' folder "
            f"(run 'dir {DATA_DIR}' on Windows or 'ls {DATA_DIR}' on Mac/Linux to confirm) "
            f"and add your 3GPP documents there first."
        )

    print(f"Loaded {len(documents)} document page(s)/file(s) from {DATA_DIR}/")
    return documents


def chunk_documents(documents):
    """Split documents into overlapping chunks.

    chunk_size=1000 characters and chunk_overlap=150 balance two failure modes:
      - chunks too large  -> irrelevant text dilutes retrieval and wastes context window
      - chunks too small  -> a clause/definition can get cut in half, losing meaning

    The overlap means that if a sentence or definition falls right on a chunk
    boundary, it still appears intact in at least one chunk.

    The `separators` list tells the splitter to prefer breaking on paragraph
    breaks first, then sentences, only falling back to raw character cuts as
    a last resort -- this keeps chunks more semantically coherent, which
    matters a lot for a structured, clause-based document like a 3GPP spec.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_index(chunks):
    """Embed every chunk and build a FAISS similarity index, saved to disk
    so app.py doesn't have to re-embed everything on every run.

    We use a small local sentence-transformer model (all-MiniLM-L6-v2) for
    embeddings rather than a paid API: it's free, runs offline, and is
    plenty accurate for semantic similarity search at this scale.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_index(chunks)