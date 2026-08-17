"""
app.py

Streamlit chat UI for the 3GPP RAG chatbot.

Flow for every question:
  1. Embed the user's question with the same embedding model used at ingest time.
  2. Retrieve the top-k most similar chunks from the FAISS index.
  3. Drop any chunk that isn't similar enough -- this is the key hallucination
     guardrail. If nothing relevant is found, we tell the user directly instead
     of forcing the LLM to invent an answer.
  4. Build a strict prompt that only allows the model to answer from the
     retrieved chunks.
  5. Call Groq for generation at low temperature, then show the answer
     alongside the exact sources it was grounded in.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

load_dotenv()

INDEX_DIR = "faiss_index"
TOP_K = 8
DISTANCE_THRESHOLD = 1.1

GROQ_MODEL = "openai/gpt-oss-120b" 

DEBUG_MODE = True


def handle_small_talk(message: str):
    """Handle simple greetings/basic conversation without querying the 3GPP index."""
    text = " ".join(message.lower().strip().split())
    responses = {
        "hi": "Hi! 👋 I’m your 3GPP Standards assistant. Ask me anything about the ingested 3GPP documents.",
        "hello": "Hello! 👋 I’m your 3GPP Standards assistant. What would you like to know?",
        "hey": "Hey! 👋 Ask me a question about the 3GPP standards whenever you’re ready.",
        "good morning": "Good morning! 👋 How can I help with the 3GPP standards?",
        "good afternoon": "Good afternoon! 👋 How can I help with the 3GPP standards?",
        "good evening": "Good evening! 👋 How can I help with the 3GPP standards?",
        "how are you": "I’m doing well! I’m ready to help you explore the 3GPP standards. 📡",
        "how are you doing": "I’m doing well! I’m ready to help you explore the 3GPP standards. 📡",
        "thanks": "You’re welcome! 😊",
        "thank you": "You’re welcome! 😊",
        "bye": "Bye! 👋 Come back whenever you have a 3GPP question.",
    }
    return responses.get(text)

SYSTEM_PROMPT = """You are a technical assistant for a 3GPP standards knowledge base.

For technical 3GPP questions, answer ONLY using the provided 3GPP standards excerpts below.

Rules:
- Answer strictly from the given context. Do not use any outside knowledge about telecom or 3GPP, even if you already know it.
- If the answer is not contained in the context, respond with exactly: "I don't have enough information in the provided documents to answer that."
- Write a clean, well-organized answer in plain prose or bullet points. Do NOT include citation labels like "(Excerpt 1)", "Source:", or filenames/page numbers inside your answer -- the sources are already shown separately to the user below your answer, so repeating them clutters the response.
- Be precise and technical. Do not speculate or fill gaps with assumptions.
"""


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def retrieve_context(vectorstore, question):
    """Return only the chunks that pass the similarity threshold, paired with their distance score."""
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    if DEBUG_MODE:
        print(f"\n[DEBUG] Question: {question}")
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  distance={score:.4f} | source={source} | preview={preview!r}")

    return [(doc, score) for doc, score in results if score <= DISTANCE_THRESHOLD]


def build_user_prompt(question, relevant_chunks):
    context_blocks = []
    for i, (doc, score) in enumerate(relevant_chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[Excerpt {i} | source: {source}, page: {page}]\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)
    return f"Context:\n{context_text}\n\nQuestion: {question}"


def ask_groq(client, question, relevant_chunks):
    


    
    if not relevant_chunks:
        return "I don't have enough information in the provided documents to answer that."

    user_prompt = build_user_prompt(question, relevant_chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,  
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="3GPP RAG Chatbot", page_icon="📡")
st.title("📡 3GPP Standards RAG Chatbot")
st.caption(
    "Answers are grounded strictly in the ingested 3GPP documents. "
    "If nothing relevant is found, the bot says so instead of guessing."
)

if not os.path.exists(INDEX_DIR):
    st.error("No FAISS index found. Run `python ingest.py` first after adding documents to data/.")
    st.stop()

vectorstore = load_vectorstore()
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the ingested 3GPP documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        small_talk_answer = handle_small_talk(question)

        if small_talk_answer is not None:
            # Greetings/basic conversation should not be treated as 3GPP retrieval queries.
            answer = small_talk_answer
            relevant_chunks = []
            st.markdown(answer)
        else:
            with st.spinner("Retrieving relevant sections..."):
                relevant_chunks = retrieve_context(vectorstore, question)
                answer = ask_groq(groq_client, question, relevant_chunks)

            st.markdown(answer)

        if relevant_chunks:
            with st.expander("View retrieved sources"):
                for i, (doc, score) in enumerate(relevant_chunks, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Excerpt {i}** — `{source}`, page {page} (distance: {score:.3f})")
                    preview = doc.page_content[:500]
                    st.text(preview + ("..." if len(doc.page_content) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})