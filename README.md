💬 LLM PDF Chat
===============

An interactive chatbot application that allows users to upload PDF documents and ask questions about their content. This project uses advanced language models and vector embeddings to provide accurate, context-based answers.

Features
--------

-   PDF Upload: Easily upload a PDF document for analysis.

-   Text Extraction: Automatically extracts text from uploaded PDFs.

-   Semantic Search: Uses FAISS for efficient vector-based similarity search.

-   Conversational AI: Powered by Google's Flan-T5 Large model for natural language responses.

-   Persistent Storage: Saves embeddings locally to avoid reprocessing the same document.

Technologies Used
-----------------

-   Streamlit: For building the user interface.

-   LangChain: Framework for integrating language models and chains.

-   FAISS: For vector-based document retrieval.

-   PyPDF2: For extracting text from PDF files.

-   HuggingFace Hub: To access pre-trained models like Flan-T5 Large.

-   Python Dotenv: For managing environment variables.

Installation
------------

Follow these steps to set up the project:

1.  Clone the repository:

    bash

    `git clone https://github.com/yourusername/llm-pdf-chat.git cd llm-pdf-chat `

2.  Install the required dependencies:

    bash

    `pip install -r requirements.txt `

3.  Create a `.env` file in the root directory and add your HuggingFace API token:

    text

    `HUGGINGFACEHUB_API_TOKEN=your_huggingface_token `

Usage
-----

1.  Run the Streamlit application:

    bash

    `streamlit run app.py `

2.  Upload a PDF document using the file uploader in the app.

3.  Wait for the app to process the document and generate embeddings.

4.  Enter your query in the text input field.

5.  View the AI-generated response based on the content of your PDF.

How It Works
------------

1.  The application extracts text from the uploaded PDF.

2.  The text is split into manageable chunks.

3.  These chunks are converted into vector embeddings using HuggingFace's sentence transformer.

4.  Embeddings are stored in a FAISS vector database for efficient retrieval.

5.  When a query is entered, the most relevant text chunks are retrieved.

6.  The Flan-T5 Large model generates a comprehensive answer based on these chunks.

Requirements
------------

-   streamlit

-   langchain

-   faiss-cpu

-   huggingface_hub

-   PyPDF2

-   python-dotenv

-   streamlit-extras

-   sentence-transformers

Author
------

👨‍💻 Made by Hemant Soni

Acknowledgements
----------------

🎥 Inspired by [Prompt Engineer - YouTube](https://www.youtube.com/@promptengineer)
