"""Standalone student assignment built from Buddy's UX and RAG concept."""
import streamlit as st
from pathlib import Path
from rag import RagEngine, answer
st.set_page_config(page_title="Buddy | RAG Lab",page_icon="🤝",layout="wide")
st.markdown("""<style>.stApp {background:linear-gradient(135deg,#101620,#1a2633);color:#eef4f8} h1,h2,h3{color:#d9f4e4} [data-testid=stSidebar]{background:#121c26}</style>""",unsafe_allow_html=True)
st.title("🤝 Buddy · RAG Lab")
st.caption("Northstar Labs · Fictional HR policies · Assignment 2: Context Engineering & RAG")
@st.cache_resource(show_spinner="Loading the embedding model and vector database...")
def get_engine():
    engine=RagEngine(documents_dir="documents",db_path="chroma_data")
    engine.ingest()
    return engine
try: engine=get_engine()
except Exception as e:
    st.error(f"Knowledge base initialization failed: {e}");st.stop()
with st.sidebar:
    st.header("Knowledge base")
    st.metric("Policy documents",len(list(Path('documents').glob('*.md'))))
    st.metric("Indexed chunks",engine.collection.count())
    st.caption("Embedding: all-MiniLM-L6-v2 · Chroma vector search · Groq generation")
    st.divider()
    st.info("Demo uses synthetic policies, not official employment advice.")
    if st.button("Rebuild document index"):
        engine.ingest(rebuild=True);st.success("Reindexed")
mode=st.radio("Answer mode",["RAG (retrieved evidence)","Pure LLM (no retrieval)","Compare side by side"],horizontal=True)
with st.expander("View source documents"):
    for item in sorted(Path('documents').glob('*.md')): st.write(item.name)
question=st.chat_input("Ask about PTO, BGV, onboarding, benefits...")
if question:
    st.chat_message("user").write(question)
    chunks=engine.retrieve(question,k=3)
    try:
        if mode=="Compare side by side":
            left,right=st.columns(2)
            with left: st.subheader("RAG");st.write(answer(question,chunks))
            with right: st.subheader("Pure LLM");st.write(answer(question,None))
        else:
            retrieved=mode.startswith("RAG")
            st.chat_message("assistant").write(answer(question,chunks if retrieved else None))
        if mode!="Pure LLM (no retrieval)":
            with st.expander("Retrieved evidence · top 3 chunks"):
                for i,c in enumerate(chunks,1):
                    st.markdown(f'**{i}. {c["source"]} · chunk {c["chunk"]} · distance {c["distance"]:.3f}**')
                    st.caption(c["text"])
    except Exception as e: st.error(f"Answer generation unavailable: {e}. Set GROQ_API_KEY in .env.")
