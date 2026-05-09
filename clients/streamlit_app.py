import os

import httpx
import streamlit as st


API_URL = os.environ.get("RAG_API_URL", "http://localhost:8080")


st.set_page_config(
    page_title="RAG Docker",
    page_icon="📄",
    layout="wide",
)

st.title("RAG Docker — Financial Report Q&A")


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox(
        "Chat Provider",
        ["openai", "anthropic", "gemini"],
        index=0,
    )
    top_k = st.slider("Top K Results", min_value=1, max_value=20,
                       value=5)
    rerank_method = st.selectbox(
        "Re-ranking",
        ["cross-encoder", "none"],
        index=0,
    )

    st.divider()
    st.header("Documents")

    if st.button("Refresh Document List"):
        st.session_state.pop("documents", None)

    if "documents" not in st.session_state:
        try:
            resp = httpx.get(f"{API_URL}/documents", timeout=10.0)
            st.session_state["documents"] = resp.json()
        except Exception:
            st.session_state["documents"] = []

    docs = st.session_state["documents"]
    if docs:
        for doc in docs:
            st.text(f"📄 {doc['file']} ({doc['chunks']} chunks)")
    else:
        st.info("No documents ingested yet.")

    st.divider()
    st.header("Ingest")

    source_dir = st.text_input("Source directory", value="docs/")
    chunking = st.selectbox(
        "Chunking method",
        ["recursive", "semantic", "by_title"],
    )

    if st.button("Ingest Documents"):
        with st.spinner("Ingesting..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/ingest",
                    json={
                        "source_dir": source_dir,
                        "chunking_method": chunking,
                    },
                    timeout=300.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Ingested {data['documents_ingested']} docs "
                        f"({data['total_chunks']} chunks)"
                    )
                    st.session_state.pop("documents", None)
                else:
                    st.error(resp.json().get("detail", "Failed"))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        status = health.get("status", "unknown")
        color = "green" if status == "healthy" else "red"
        st.markdown(
            f"**Status:** :{color}[{status}] | "
            f"**ChromaDB:** {health.get('chromadb', '?')}"
        )
    except Exception:
        st.markdown("**Status:** :red[API offline]")


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['file']}** (p.{src['page']}) "
                        f"— {src['section']}"
                    )
                    st.caption(src["excerpt"][:200])

if prompt := st.chat_input("Ask about the financial reports..."):
    st.session_state["messages"].append({
        "role": "user", "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "chat_provider": provider,
                        "top_k": top_k,
                        "rerank_method": rerank_method,
                    },
                    timeout=120.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])

                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(
                                    f"**{src['file']}** "
                                    f"(p.{src['page']}) "
                                    f"— {src['section']}"
                                )
                                st.caption(
                                    src["excerpt"][:200]
                                )

                    meta = data.get("metadata")
                    if meta:
                        st.caption(
                            f"{meta['provider']}/{meta['model']} "
                            f"| {meta['retrieval_count']} chunks "
                            f"| {meta['latency_ms']}ms"
                        )

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": sources,
                    })
                else:
                    error = resp.json().get("detail", "Query failed")
                    st.error(error)
            except Exception as e:
                st.error(f"Error: {e}")
