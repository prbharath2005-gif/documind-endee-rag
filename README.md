\# 📄 DocuMind: Semantic Document Search \& RAG Powered by Endee



\[!\[Endee DB](https://img.shields.io/badge/Vector%20DB-Endee-blue)](https://github.com/endee-io/endee)

\[!\[Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

\[!\[Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



DocuMind is an intelligent document search and Retrieval-Augmented Generation (RAG) assistant. It parses unstructured PDF documents, converts text chunks into 384-dimensional dense vector representations, indexes them into the \*\*Endee Vector Database\*\*, and performs high-speed semantic similarity searches to retrieve contextually relevant document snippets.



\---



\## 📌 Problem Statement

Traditional keyword-based document searching relies on exact string matches, frequently missing relevant context when users query using different vocabulary, synonyms, or natural language phrasing. In dense documents like technical manuals, research papers, or legal agreements, key information remains hidden behind phrasing variations. 



\*\*DocuMind\*\* solves this by converting document contents and user queries into high-dimensional vector embeddings, allowing true \*\*semantic context matching\*\* powered by \*\*Endee Vector DB\*\*.



\---



\## 🏗️ System Design \& Technical Approach



```text

┌─────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐

│  Upload PDF     │ ───> │ Text Chunking \&         │ ───> │ Endee Vector Database  │

│  (PyPDF)        │      │ Sentence Transformers   │      │ (384-dim Indexing)     │

└─────────────────┘      └─────────────────────────┘      └────────────────────────┘

&#x20;                                                                      │

┌─────────────────┐      ┌─────────────────────────┐                   │

│  User Semantic  │ ───> │ Vectorize Query \&       │ <─────────────────┘

│  Search Query   │      │ Similarity Match        │   Cosine Vector Search

└─────────────────┘      └─────────────────────────┘

