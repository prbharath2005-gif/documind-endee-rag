import streamlit as st
from pypdf import PdfReader
from src.embeddings import EmbeddingEngine
from src.database import EndeeStore

st.set_page_config(page_title="DocuMind - Endee RAG", layout="wide")
st.title("📄 DocuMind: Semantic Document Search")

@st.cache_resource
def load_services():
    return EmbeddingEngine(), EndeeStore()

embedder, db = load_services()

# Sidebar: Document Processing
st.sidebar.header("1. Upload & Index")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and st.sidebar.button("Index Document"):
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    # Simple chunking logic (500 characters per chunk)
    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500) if raw_text[i:i+500].strip()]
    
    # Generate embeddings
    st.sidebar.info("Generating embeddings...")
    embeddings = embedder.encode_texts(chunks)
    
    # Prepare IDs and Payloads
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    payloads = [{"text": chunk, "source": uploaded_file.name} for chunk in chunks]

    # Upsert into Endee
    db.add_documents(ids, embeddings, payloads)
    st.sidebar.success(f"Successfully indexed {len(chunks)} chunks into Endee!")

# Main Area: Querying
st.header("2. Ask Questions")
user_query = st.text_input("Enter your search query:")

if user_query:
    query_vector = embedder.encode_query(user_query)
    results = db.query_similar(query_vector, top_k=3)

    st.subheader("Top Context Match Results:")
    for res in results:
        with st.expander(f"Match Score / Payload ID: {getattr(res, 'id', 'Result')}"):
            st.write(res.payload.get("text", "No text payload found"))