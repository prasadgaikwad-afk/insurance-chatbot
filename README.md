# Insurance Policy Comparison Chatbot for HDFC Life and ICICI Lombard

## Overview

This Streamlit application is designed to compare two insurance policy documents uploaded as PDF files. It leverages LangChain for document processing, Hugging Face Transformers for embeddings, FAISS for vector storage, and Groq's Gemma-7B-IT model for generating comparative summaries.

## Features

-   **PDF Upload and Processing:** Accepts two PDF files, extracts text, and prepares it for analysis.
-   **Text Cleaning and Summarization:** Cleans extracted text and generates concise summaries using Groq's Gemma model.
-   **Vector Storage and Retrieval:** Stores processed text embeddings in a FAISS vector database for efficient semantic search.
-   **Query-Based Comparison:** Allows users to ask questions about the policies and retrieves relevant information from the documents.
-   **Comparative Response Generation:** Generates a detailed comparative analysis using Groq's Gemma model, presenting the information in a structured format.
-   **Clear Output:** Displays the user's question, the generated response, and the names of the uploaded PDF files.

## Setup and Installation

1.  **Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Obtain Groq API Key:**
    -   Sign up for a Groq account and obtain your API key.
    -   Replace with your actual Groq API key in the `GROQ_API_KEY` variable within the script.
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Upload PDF Files:**
    -   Use the file uploader to upload two PDF files containing the insurance policy documents.
2.  **Process PDFs and Start Chat:**
    -   Click the "Process PDFs and Start Chat" button.
    -   The application will process the PDFs, create embeddings, and prepare the chatbot.
3.  **Ask Questions:**
    -   Enter your questions about the insurance policies in the chat input box.
    -   The chatbot will retrieve relevant information and generate a comparative response.
4.  **View Results:**
    -   The application will display your question, the generated response, and the names of the uploaded PDF files.

## Code Structure

-   `process_pdfs(pdf_files)`: Processes the uploaded PDF files and extracts text.
-   `clean_text(text)`: Cleans the extracted text by removing unnecessary characters and whitespace.
-   `summarize_text(text)`: Summarizes the text using Groq's Gemma model.
-   `search_query(query, vector_db, embedding_model, k=5)`: Searches the vector database for relevant information.
-   `generate_gemma_comparison(query, csv_file="summarized_results.csv")`: Generates a comparative response using Groq's Gemma model.
-   `main()`: Main function that sets up the Streamlit application and handles user interactions.


## Workflow

![Workflow](https://github.com/user-attachments/assets/cac679ce-dd7d-4640-8e29-da8cce1fda68)
