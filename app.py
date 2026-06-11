import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from query_hybrid import hybrid_search

# --- AUTOMATIC BACKGROUND INITIALIZATION ---
# This forces the cloud server to read embed_books.py and index your files if qdrant is empty
@st.cache_resource
def initialize_database_on_cloud():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", location=":memory:")
        collections = client.get_collections().collections
        exists = any(c.name == "space_books" for c in collections)
        
        if not exists:
            st.toast("?? Initializing Cloud Vector Index...")
            import subprocess
            subprocess.run(["python", "embed_books.py"], capture_output=True)
            st.toast("? Vector Database Loaded!")
    except Exception as e:
        pass

initialize_database_on_cloud()

# --- STREAMLIT UI ---
st.title("?? Space Books Hybrid Explorer")
st.write("Search space stories using BM25 keyword matching fused with dense Qdrant vector embeddings")

query = st.text_input("e.g. NASA mission studying liquid water on Saturn or a moon...", "")

if st.button("Search Database"):
    if query:
        with st.spinner("Analyzing semantic structures..."):
            results = hybrid_search(query)
            if results:
                for doc, score in results:
                    st.markdown(f"### ?? {doc['title']}")
                    st.write(doc['text'])
                    st.caption(f"Blended Score: {score}")
                    st.write("---")
            else:
                st.info("No matching stories found for that specific search.")
    else:
        st.warning("Please enter a query first!")
