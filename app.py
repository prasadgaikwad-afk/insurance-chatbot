import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
from transformers import pipeline
import csv
from groq import Groq
import tempfile

import re
import textwrap

# Replace with your Groq API key
GROQ_API_KEY = "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"
FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.csv"

def store_embeddings(chunks, embedding_model, doc_sources):
    """Stores text chunks and their embeddings in FAISS and metadata in CSV."""
    embeddings = embedding_model.embed_documents(chunks)
    vector_dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'content'])
        for source, content in zip(doc_sources, chunks):
            writer.writerow([source, content])

def process_pdfs(pdf_files):
    """Processes uploaded PDF files and returns their processed text."""
    processed_texts = []
    for pdf_file in pdf_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(pdf_file.read())
                temp_pdf_path = temp_pdf.name

            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separator="\n\n")
            texts = text_splitter.split_documents(documents)
            page_contents = [doc.page_content for doc in texts]
            combined_text = "\n\n".join(page_contents)
            processed_texts.append(combined_text)
            os.remove(temp_pdf_path)
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
            processed_texts.append("")
    return processed_texts

def clean_text(text):
    """Enhanced text cleaning function."""
    if not text:
        return ""
    text = " ".join(text.split())
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'\s*•\s*', '. ', text)
    text = re.sub(r'\s*–\s*', '-', text)
    return text.strip()

def summarize_text(text):
    """Summarizes text using Groq's Gemma-7B-IT model."""
    if not text:
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    max_input_length = 4096
    input_length = len(text.split())

    def get_summary(chunk):
        prompt = f"""
        [INST] Summarize the following text while preserving key details and clarity.

        Text: {chunk}

        Summary: [/INST]
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it"
        )
        return chat_completion.choices[0].message.content.strip()

    if input_length > max_input_length:
        chunks = textwrap.wrap(text, width=max_input_length)
        summaries = [get_summary(chunk) for chunk in chunks]
        return " ".join(summaries)
    else:
        return get_summary(text)

def load_embeddings_and_search(query, embedding_model, k=5):
    """Loads embeddings from FAISS and performs a search."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.warning("FAISS index file not found. Please upload and process PDFs first.")
        return []
    if not os.path.exists(METADATA_PATH):
        st.warning("Metadata file not found. Please upload and process PDFs first.")
        return []

    index = faiss.read_index(FAISS_INDEX_PATH)
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)

    results = []
    with open(METADATA_PATH, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        all_data = list(reader)
        for i in indices[0]:
            if i < len(all_data):
                results.append((all_data[i][0], all_data[i][1], distances[0][list(indices[0]).index(i)]))
    return results

def generate_gemma_comparison(query, results, max_context_length=1500):
    """Generates a comparison response using Gemma2-9b-it based on retrieved data, with length constraints."""
    pdf1_results = [result[1] for result in results if result[0] == 'pdf1']
    pdf2_results = [result[1] for result in results if result[0] == 'pdf2']

    def truncate_or_summarize(text_list):
        processed_texts = []
        for text in text_list:
            if len(text) > max_context_length:
                processed_texts.append(summarize_text(text[:max_context_length]))
            else:
                processed_texts.append(text)
        return processed_texts

    pdf1_results = truncate_or_summarize(pdf1_results)
    pdf2_results = truncate_or_summarize(pdf2_results)

    pdf1_context = "\n".join(pdf1_results) if pdf1_results else "No relevant information found in the first uploaded document."
    pdf2_context = "\n".join(pdf2_results) if pdf2_results else "No relevant information found in the second uploaded document."

    prompt = f"""
    [INST] You are an expert insurance comparison assistant. Given the following information from two policy documents, analyze and compare their coverage, exclusions, pricing, and additional benefits based on the user's question.

    Document 1 Information:
    {pdf1_context}

    Document 2 Information:
    {pdf2_context}

    User Question: {query}

    **Step-by-Step Analysis (Chain of Thought Technique):**
    1. **Identify Relevant Information**: Find the sections in each document that address the user's question.
    2. **Compare Coverage**: Detail the coverage offered by each policy related to the query.
    3. **Highlight Differences**: Point out any significant differences or advantages of one policy over the other.
    4. **Summarize**: Provide a concise summary answering the user's question based on the provided information.

    Based on this structured approach, generate a detailed comparative response.
    Comparison: [/INST]
    """

    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="gemma2-9b-it")
        response = chat_completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        return f"Error generating response from Groq: {e}"

def main():
    st.title("Insurance Policy Comparison Chatbot")
    pdf_files = st.file_uploader("Upload two PDF files", type="pdf", accept_multiple_files=True)

    if pdf_files and len(pdf_files) == 2:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            if st.button("Process PDFs and Start Chat"):
                with st.spinner("Processing PDFs and creating embeddings..."):
                    pdf_texts = process_pdfs(pdf_files)
                    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    doc_sources = []
                    chunks = []

                    for idx, text in enumerate(pdf_texts):
                        if text:
                            pdf_chunks = text_splitter.split_text(text)
                            chunks.extend(pdf_chunks)
                            doc_sources.extend([f"pdf{idx+1}"] * len(pdf_chunks))

                    store_embeddings(chunks, embedding_model, doc_sources)
                    st.session_state.pdf_names = [pdf_files[0].name, pdf_files[1].name]
                    st.write("Chatbot is ready. Ask your questions!")
        else:
            st.session_state.pdf_names = [pdf_files[0].name, pdf_files[1].name]
            st.write("Chatbot is ready. Ask your questions!")

        if 'pdf_names' in st.session_state:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            query = st.chat_input("Ask a question about the insurance policies:")
            if query:
                retrieved_docs = load_embeddings_and_search(query, embedding_model)
                if retrieved_docs:
                    response = generate_gemma_comparison(query, retrieved_docs)
                    st.write(f"**Question:** {query}")
                    st.write(f"**Answer:** {response}")
                    st.write(f"**Policy 1: {st.session_state.pdf_names[0]} (Document 1)**")
                    st.write(f"**Policy 2: {st.session_state.pdf_names[1]} (Document 2)**")
                else:
                    st.warning("No relevant information found in the processed documents for your query.")
    elif pdf_files:
        st.warning("Please upload exactly two PDF files.")
    else:
        st.info("Please upload two PDF files to start.")

if __name__ == "__main__":
    main()
