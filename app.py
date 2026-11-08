import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

from rag import answer, build_index, load_docs, collection

@st.cache_resource
def init_index():
    if collection.count() == 0:
        build_index(load_docs())
    return True

init_index()

st.set_page_config(page_title="ML Textbook RAG Q&A", page_icon="📚")
st.title("📚 ML Textbook RAG Q&A")
st.caption("Over study notes on Zhou Zhihua's *Machine Learning* · "
           "two-stage retrieval (embedding recall + rerank) + Claude · "
           "knowledge base is in Chinese")

q = st.text_input("Ask a question about machine learning:",
                  placeholder="e.g. 决策树为什么要剪枝？")

if q:
    with st.spinner("Retrieving + generating…"):
        ans, hits = answer(q)
    st.markdown("### Answer")
    st.markdown(ans)
    with st.expander("📎 Sources (retrieved chunks)"):
        for i, (chunk, score) in enumerate(hits, 1):
            st.markdown(f"**Chunk {i}** · relevance {score:.3f}")
            st.text(chunk[:500])