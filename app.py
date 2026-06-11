import streamlit as st
import sys

# Import search resources from query_hybrid.py
try:
    from query_hybrid import (
        books,
        books_by_id,
        bm25,
        client,
        model,
        tokenize,
        collection_name,
    )
except ImportError:
    st.error("Could not import resources from query_hybrid.py. Please make sure the file exists in the directory.")
    sys.exit(1)

# Configure Streamlit page
st.set_page_config(
    page_title="Space Books Hybrid Search",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Premium Space Dark Theme & Glassmorphic Cards
st.markdown(
    """
    <style>
        /* Force styling for the cards and typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        .main-header {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .sub-header {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 40px;
        }

        .space-card {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(99, 102, 241, 0.15);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .space-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 12px 30px rgba(99, 102, 241, 0.2);
            background: rgba(30, 41, 59, 0.85);
        }
        
        .card-index {
            font-size: 0.9rem;
            font-weight: 700;
            color: #818cf8;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .card-title {
            color: #f8fafc !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            margin-bottom: 12px !important;
            font-family: 'Inter', sans-serif;
            line-height: 1.3;
        }
        
        .card-content {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 20px;
            font-family: 'Inter', sans-serif;
        }
        
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 14px;
        }
        
        .metric-badge {
            font-size: 0.8rem;
            padding: 5px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            display: inline-flex;
            align-items: center;
        }
        
        .badge-rrf {
            background: rgba(99, 102, 241, 0.15);
            color: #c7d2fe;
            border: 1px solid rgba(99, 102, 241, 0.30);
        }
        
        .badge-bm25 {
            background: rgba(234, 179, 8, 0.12);
            color: #fef08a;
            border: 1px solid rgba(234, 179, 8, 0.25);
        }
        
        .badge-semantic {
            background: rgba(34, 197, 94, 0.12);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.25);
        }
        
        /* Stylize default streamlit elements */
        div[data-baseweb="input"] {
            border-radius: 10px;
            border-color: rgba(99, 102, 241, 0.2);
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.5rem 2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Application Header
st.markdown("<h1 class='main-header'>🛰️ Space Books Hybrid Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Search space stories using BM25 keyword matching fused with dense Qdrant vector embeddings</p>", unsafe_allow_html=True)

# Main search form layout
with st.container():
    query_input = st.text_input(
        label="Search Database",
        placeholder="e.g. NASA mission studying liquid water on Saturn or a moon...",
        label_visibility="collapsed",
    )
    
    # Center the search button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_clicked = st.button("Search Database", type="primary", use_container_width=True)

# Fusion function definition
def run_hybrid_search(query, k=60, top_n=50):
    if not query.strip():
        return []

    # 1. Lexical BM25 Scoring
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        [(idx, score) for idx, score in enumerate(bm25_scores) if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    # 2. Dense Semantic Vector Retrieval
    query_vector = model.encode(query).tolist()
    semantic_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_n,
    )
    semantic_points = semantic_response.points

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    bm25_ranks = {}
    semantic_ranks = {}

    # Accumulate ranks for BM25
    for rank, (doc_idx, score) in enumerate(bm25_ranked[:top_n], start=1):
        book_id = books[doc_idx]["id"]
        rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + rank))
        bm25_ranks[book_id] = rank

    # Accumulate ranks for Semantic Search
    for rank, point in enumerate(semantic_points, start=1):
        book_id = point.id
        rrf_scores[book_id] = rrf_scores.get(book_id, 0.0) + (1.0 / (k + rank))
        semantic_ranks[book_id] = rank

    # Sort candidates by combined RRF score
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Map back to story objects
    results = []
    for book_id, rrf_score in sorted_rrf[:3]:
        book = books_by_id.get(book_id)
        if book:
            results.append({
                "id": book_id,
                "title": book.get("title", ""),
                "content": book.get("content", ""),
                "rrf_score": rrf_score,
                "bm25_rank": bm25_ranks.get(book_id, "N/A"),
                "semantic_rank": semantic_ranks.get(book_id, "N/A"),
            })
    return results

# Process Query
if search_clicked or (query_input and st.session_state.get('last_query') != query_input):
    st.session_state['last_query'] = query_input
    
    if not query_input.strip():
        st.warning("Please enter a valid query string.")
    else:
        with st.spinner("Executing hybrid fusion search query..."):
            results = run_hybrid_search(query_input)
            
        if not results:
            st.info("No matching records found. Try simplifying or using different search terms.")
        else:
            st.markdown(f"### Top 3 Combined Results for: *\"{query_input}\"*")
            
            for idx, res in enumerate(results, start=1):
                bm25_val = f"Rank {res['bm25_rank']}" if res['bm25_rank'] != "N/A" else "Not Ranked"
                semantic_val = f"Rank {res['semantic_rank']}" if res['semantic_rank'] != "N/A" else "Not Ranked"
                
                # HTML Card generation
                card_html = f"""
                <div class="space-card">
                    <div class="card-index">Rank #{idx} in Combined Fusion</div>
                    <div class="card-title">{res['title']}</div>
                    <div class="card-content">{res['content']}</div>
                    <div class="metrics-container">
                        <span class="metric-badge badge-rrf">📊 RRF Score: {res['rrf_score']:.5f}</span>
                        <span class="metric-badge badge-bm25">🔍 Lexical (BM25): {bm25_val}</span>
                        <span class="metric-badge badge-semantic">🧠 Semantic (Qdrant): {semantic_val}</span>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
