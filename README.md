📡 3GPP Standards RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers technical questions using an ingested 3GPP standards knowledge base.

The main goal is to keep answers grounded in the retrieved 3GPP content and reduce hallucinations by filtering retrieval results before sending context to the LLM.

Features

RAG-based question answering over 3GPP standards

Local embeddings using sentence-transformers/all-MiniLM-L6-v2

Local FAISS vector index for similarity search

Groq LLM for answer generation

Strict context-based prompting

Similarity-distance threshold to reject weak retrieval results

Source/document and page information shown with retrieved excerpts

Simple greeting handling (hi, hello, hey, etc.) without querying the 3GPP index

Streamlit chat interface

Knowledge Base

The intended knowledge base contains these 3GPP Stage 2 specifications:

TS 23.501 — System Architecture for the 5G System (5GS)

TS 23.502 — Procedures for the 5G System (5GS)

TS 23.503 — Policy and Charging Control Framework for the 5G System (5GS)

The project uses matching Release 20 versions of these specifications.

Architecture

                 3GPP Documents
                       │
                       ▼
                Document Loading
                       │
                       ▼
                    Chunking
              (1000 chars / 150 overlap)
                       │
                       ▼
                   Embeddings
          all-MiniLM-L6-v2 (local)
                       │
                       ▼
                 FAISS Vector Index
                       │
                       │
        ┌──────────────┘
        │
        ▼
    User Question
        │
        ├── Simple greeting? ──► Direct response
        │
        ▼
   Vector Similarity Search
        │
        ▼
      Top-K Chunks
        │
        ▼
 Similarity Distance Threshold
        │
        ├── No relevant chunks ──► Refuse / insufficient information
        │
        ▼
  Strict Context + Question Prompt
        │
        ▼
      Groq LLM
        │
        ▼
   Grounded Answer
        │
        ▼
 Retrieved Sources / Pages

Tech Stack

Python

Streamlit — web UI

LangChain Community — document loading and vector-store integration

LangChain Text Splitters — chunking

FAISS — vector similarity search

Sentence Transformers — local embeddings

Groq — LLM inference

python-dotenv — environment variable loading

Project Structure

3gpp-rag-chatbot/
│
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
│
├── data/
│   └── 3GPP source documents
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl

ingest.py

Responsible for:

Loading PDF, TXT, and DOCX files from data/

Splitting documents into overlapping chunks

Creating embeddings

Building the FAISS index

Saving the index locally

Run it whenever the source documents change.

app.py

Responsible for:

Loading the FAISS index

Receiving user questions

Handling simple greetings separately

Retrieving relevant chunks

Applying the similarity threshold

Building the grounded prompt

Calling the Groq model

Displaying the answer and retrieved sources

Setup

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd 3gpp-rag-chatbot

2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Add the 3GPP source documents

Place the source documents inside:

data/

The ingestion script supports:

.pdf

.txt

.docx

5. Create the FAISS index

python ingest.py

You should see output showing the number of documents loaded, chunks created, and the index saved.

6. Configure the Groq API key

Create a .env file locally:

GROQ_API_KEY=your_groq_api_key

Do not commit .env to GitHub.

7. Run the application

streamlit run app.py

Retrieval and Hallucination Controls

The application uses multiple safeguards:

1. Similarity filtering

The application retrieves the nearest chunks and only keeps chunks whose FAISS distance is within the configured threshold.

Current configuration:

TOP_K = 1
DISTANCE_THRESHOLD = 1.1

2. Strict grounding prompt

The LLM is instructed to answer technical questions only from the retrieved 3GPP context and to refuse when the required information is not present.

3. No-context refusal

If no retrieved chunk passes the threshold, the application does not call the LLM and returns:

I don't have enough information in the provided documents to answer that.

4. Low-temperature generation

The Groq model is called with:

temperature=0.1

This is intended to reduce unnecessary randomness in technical answers.

Example Questions

What is the role of the AMF?

What is the role of the SMF?

What is the relationship between AMF and SMF during PDU Session Establishment?

What is PDU Session Establishment?

What is the purpose of the N2 interface?

What is 5QI?

Out-of-scope examples

What is the capital of France?

Who invented Facebook?

These should not be answered from external knowledge. The application is designed to reject questions when relevant information is not available in the 3GPP knowledge base.

Known Limitations

This is a prototype RAG system and retrieval quality can vary with question phrasing.

For example, semantically similar questions can produce different nearest-neighbor rankings. This can happen because the current retriever uses dense vector similarity without a second-stage re-ranker.

Potential future improvements include:

Hybrid BM25 + vector search

Cross-encoder re-ranking

Clause-aware chunking based on 3GPP section numbers

A dedicated 3GPP domain/intent classifier

A fixed evaluation dataset for retrieval and answer regression testing

Better threshold calibration using in-domain and out-of-domain test questions

Deployment

For Streamlit Community Cloud:

Push the application code and required dependencies to GitHub.

Ensure the FAISS index required by app.py is available to the deployed application.

Create the Streamlit app using app.py as the entrypoint.

Add the Groq API key through Streamlit Secrets instead of committing it to the repository.

Example Streamlit secret:

GROQ_API_KEY = "your_groq_api_key"

Security Notes

Never commit API keys or .env files.

The FAISS index is expected to be generated from trusted local project data.

allow_dangerous_deserialization=True is used when loading the locally generated FAISS index. Do not blindly load an index obtained from an untrusted source.

Project Goal

The project demonstrates a practical RAG pipeline for technical telecom documentation:

Documents
→ Chunking
→ Embeddings
→ Vector Search
→ Retrieval Filtering
→ Grounded Generation
→ Source Traceability

The focus is not only on generating an answer, but on keeping technical responses tied to the supplied 3GPP knowledge base.
