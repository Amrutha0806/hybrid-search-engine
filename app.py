import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from query_hybrid import hybrid_search

st.title("?? Space Books Hybrid Explorer")
st.write("Search space stories using BM25 keyword matching fused with dense Qdrant vector embeddings")

query = st.text_input("e.g. NASA mission studying liquid water on Saturn or a moon...", "")

if st.button("Search Database"):
    if query:
        results = hybrid_search(query)
        for doc, score in results:
            st.markdown(f"### ?? {doc['title']}")
            st.write(doc['text'])
            st.caption(f"Blended Score: {score}")
            st.write("---")
    else:
        st.warning("Please enter a query first!")
