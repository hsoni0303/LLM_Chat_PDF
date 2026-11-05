import hashlib

import streamlit as st
from dotenv import load_dotenv

from rag import LLM_MODEL, PDFIndex, answer, load_pdf

load_dotenv()
st.set_page_config(page_title="Chat with your PDF", page_icon="💬")


@st.cache_resource(show_spinner="Reading and embedding PDF…")
def build_index(digest, data):
    # Cached by content hash, so re-uploading the same PDF is instant.
    import io
    return PDFIndex(load_pdf(io.BytesIO(data)))


with st.sidebar:
    st.title("💬 LLM PDF Chat")
    st.markdown(
        f"Retrieval-augmented QA over your PDF.\n\n"
        f"- Embeddings: `all-MiniLM-L6-v2`\n- LLM: `{LLM_MODEL}`"
    )
    pdf = st.file_uploader("Upload a PDF", type="pdf")

st.header("Chat with your PDF")
if not pdf:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()

data = pdf.getvalue()
index = build_index(hashlib.sha256(data).hexdigest(), data)
if not index.chunks:
    st.error("No extractable text found (is this a scanned PDF?).")
    st.stop()

history = st.session_state.setdefault("history", [])
for role, text in history:
    st.chat_message(role).markdown(text)

if question := st.chat_input("Ask a question about the PDF"):
    st.chat_message("user").markdown(question)
    hits = index.search(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = answer(question, hits)
        st.markdown(reply)
        with st.expander("Sources"):
            for c in hits:
                st.caption(f"Page {c.page}")
                st.text(c.text[:500])
    history += [("user", question), ("assistant", reply)]
