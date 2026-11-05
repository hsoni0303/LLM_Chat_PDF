"""Retrieval-augmented QA over a single PDF: extract -> chunk -> embed -> retrieve -> answer."""

import os
from dataclasses import dataclass

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")


@dataclass
class Chunk:
    text: str
    page: int


def chunk_text(text, page, size=1000, overlap=200):
    """Split text into overlapping windows, breaking on whitespace where possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size - overlap, end)
            end = space if space > start else end
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, page))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf(file):
    from pypdf import PdfReader

    chunks = []
    for page_no, page in enumerate(PdfReader(file).pages, start=1):
        chunks += chunk_text(page.extract_text() or "", page_no)
    return chunks


class PDFIndex:
    """In-memory vector index. A single PDF is small enough that exact
    cosine search with numpy is instant, so no vector DB is needed."""

    def __init__(self, chunks):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.vectors = self.encoder.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query, k=4):
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        top = np.argsort(self.vectors @ q)[::-1][:k]
        return [self.chunks[i] for i in top]


PROMPT = """Answer the question using only the context from the PDF below.
If the answer is not in the context, say you don't know. Cite pages like [p. 3].

{context}

Question: {question}"""


def answer(question, chunks, client=None):
    if client is None:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    context = "\n\n".join(f"[p. {c.page}] {c.text}" for c in chunks)
    response = client.chat_completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content
