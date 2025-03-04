import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

with st.sidebar:
    st.title("💬 LLM PDF Chat")
    st.markdown("""
    ## About
    This app is a PDF-based chatbot powered by:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [FAISS](https://faiss.ai/) for vector search
    - [Google Flan-T5 Large](https://huggingface.co/google/flan-t5-large) for LLM responses  
    """)
    add_vertical_space(2)
    st.write("👨‍💻 Made by Hemant Soni")
    st.write("🎥 Inspired by [Prompt Engineer - YouTube](https://www.youtube.com/@promptengineer)")

def main():
    st.header("Chat with your PDF 💬")

    # PDF Upload
    pdf = st.file_uploader("📄 Upload a PDF", type='pdf')

    if pdf:
        with st.spinner("Processing PDF... ⏳"):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            store_name = pdf.name[:-4]

            # Load or create embeddings
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", 'rb') as f:
                    vectorStore = pickle.load(f)
                st.success("🔍 Embeddings Loaded Successfully!")
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                vectorStore = FAISS.from_texts(chunks, embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vectorStore, f)
                st.success("✅ Embeddings Computed and Saved!")

        # User Query Input
        query = st.text_input("🔎 Ask a question about the PDF:")

        if query:
            with st.spinner("Fetching answer... 🤖"):
                docs = vectorStore.similarity_search(query=query)

                # Use Google Flan-T5 Large from Hugging Face
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 250}
                )
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)

                # Display response
                st.markdown("### 🤖 LLM Response:")
                st.success(response)

if __name__ == '__main__':
    main()
