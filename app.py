import streamlit as st
from pypdf import PdfReader
from src.embeddings import EmbeddingEngine
from src.database import EndeeStore

st.set_page_config(page_title="DocuMind - Endee RAG", layout="wide")
st.title("📄 DocuMind: Semantic Document Search")

@st.cache_resource
def load_services():
    with st.spinner("Loading AI embedding model & Endee database..."):
        return EmbeddingEngine(), EndeeStore()

embedder, db = load_services()

# Sidebar: Document Upload & Indexing
st.sidebar.header("1. Upload & Index")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and st.sidebar.button("Index Document"):
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    # Chunk text into 500-character segments
    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500) if raw_text[i:i+500].strip()]
    
    if chunks:
        st.sidebar.info("Generating vector embeddings...")
        embeddings = embedder.encode_texts(chunks)
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        payloads = [{"text": chunk, "source": uploaded_file.name} for chunk in chunks]

        db.add_documents(ids, embeddings, payloads)
        st.sidebar.success(f"Successfully indexed {len(chunks)} chunks into Endee!")
    else:
        st.sidebar.warning("No readable text found in PDF.")

# Main Area: Querying
st.header("2. Ask Questions")
user_query = st.text_input("Enter your search query:")

if user_query:
    query_vector = embedder.encode_query(user_query)
    results = db.query_similar(query_vector, top_k=3)

    st.subheader("Top Context Match Results:")
    for item in results:
        # Extract fields whether returned as dict or object
        if isinstance(item, dict):
            item_id = item.get("id", "Result")
            score = item.get("score", item.get("distance", None))
            meta = item.get("meta", item.get("payload", {}))
        else:
            item_id = getattr(item, "id", "Result")
            score = getattr(item, "score", getattr(item, "distance", None))
            meta = getattr(item, "meta", getattr(item, "payload", {}))

        text_content = meta.get("text", str(meta)) if isinstance(meta, dict) else str(meta)
        header_text = f"Match ID: {item_id}" + (f" (Score: {score:.4f})" if isinstance(score, (int, float)) else "")
        
        with st.expander(header_text):
            st.write(text_content)