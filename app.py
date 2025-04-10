import streamlit as st
import ollama
import os
import tempfile
import numpy as np
import time
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import base64

# Initialize session state variables
DEFAULT_STATE = {
    'docs': [],
    'chat_history': [],
    'vectorizer': None,
    'vectors': None,
    'chunks': [],
    'images': [],
    'current_image': None,
    'image_names': set()
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# File Processing Functions
def process_image(image_file):
    if image_file.name in st.session_state.image_names:
        return True
    try:
        image = Image.open(image_file).convert('RGB')
        st.session_state.current_image = image
        st.session_state.images.append({'name': image_file.name, 'image': image})
        st.session_state.image_names.add(image_file.name)
        return True
    except Exception as e:
        st.error(f"Failed to process image: {str(e)}")
        return False

def extract_text_from_file(uploaded_file):
    file_handlers = {
        'pdf': lambda path: '\n'.join(p.extract_text() for p in PdfReader(path).pages if p.extract_text()),
        'docx': lambda path: '\n'.join(p.text for p in Document(path).paragraphs),
        'txt': lambda path: open(path, 'r', encoding='utf-8').read(),
        'csv': lambda path: pd.read_csv(path).to_string(index=False),
        'xlsx': lambda path: pd.read_excel(path).to_string(index=False)
    }
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type not in file_handlers:
        return "Unsupported file type"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            return file_handlers[file_type](temp_file.name)
    except Exception as e:
        return f"Error processing {uploaded_file.name}: {str(e)}"
    finally:
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

def chunk_text(text, chunk_size=500, overlap=50):
    if not text:
        return []
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def process_documents():
    if not st.session_state.docs:
        return
    text = '\n\n'.join(doc['text'] for doc in st.session_state.docs)
    st.session_state.chunks = chunk_text(text)
    if st.session_state.chunks:
        if not st.session_state.vectorizer:
            st.session_state.vectorizer = TfidfVectorizer(lowercase=True)
        st.session_state.vectors = st.session_state.vectorizer.fit_transform(st.session_state.chunks)

def get_relevant_chunks(query, top_k=3):
    if st.session_state.vectors is None or not st.session_state.chunks:
        return []
    query_vector = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, st.session_state.vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [st.session_state.chunks[i] for i in top_indices]

# Response Generation Functions
def generate_response(query):
    query = query.strip().lower()
    if st.session_state.current_image and any(word in query for word in {'image', 'picture', 'photo'}):
        return analyze_image(query)
    elif st.session_state.docs:
        return generate_document_response(query)
    return generate_chat_response(query)

def generate_document_response(query):
    chunks = get_relevant_chunks(query)
    context = "\n\n".join(chunks) if chunks else "No relevant information found."
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    try:
        response = ollama.chat(model="gemma3:12b", messages=[{"role": "user", "content": prompt}])
        return response.get('message', {}).get('content', "Failed to get response")
    except Exception as e:
        return f"Error: {str(e)}"

def generate_chat_response(query):
    # Limit chat history to the last 5 messages for faster processing
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history[-5:]]
    messages.append({"role": "user", "content": query})
    try:
        response = ollama.chat(model="gemma3:12b", messages=messages)
        return response.get('message', {}).get('content', "Failed to get response")
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_image(query):
    if not st.session_state.current_image:
        return "Please upload and select an image to analyze."
    try:
        img_buffer = io.BytesIO()
        st.session_state.current_image.save(img_buffer, format='PNG')
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        prompt = (
            "Analyze this image and:\n"
            "1. Extract any visible text\n"
            "2. Describe key visual elements\n"
            "3. Answer this question: {query}")
        response = ollama.chat(
            model="llama3.2-vision:latest",
            messages=[{
                "role": "user",
                "content": prompt.format(query=query),
                "images": [img_data]
            }])
        return response.get('message', {}).get('content', "Failed to analyze the image. Please try again.")
    except Exception as e:
        return f"Image analysis failed: {str(e)}"

# UI Components
st.title("Document Q&A Bot")
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf', 'docx', 'txt', 'csv', 'xlsx'])
    if uploaded_files:
        with st.spinner("Processing files..."):
            processed_files = st.session_state.image_names.union({doc['name'] for doc in st.session_state.docs})
            new_files = [f for f in uploaded_files if f.name not in processed_files]
            for file in new_files:
                file_ext = file.name.split('.')[-1].lower()
                if file_ext in {'png', 'jpg', 'jpeg'}:
                    if process_image(file):
                        st.success(f"✅ Processed {file.name}")
                else:
                    text = extract_text_from_file(file)
                    if text != "Unsupported file type":
                        st.session_state.docs.append({'name': file.name, 'text': text})
                        st.success(f"✅ Processed {file.name}")
            if new_files:
                process_documents()
    if st.button("Clear All Files"):
        for key in DEFAULT_STATE:
            st.session_state[key] = DEFAULT_STATE[key]
        st.success("Files cleared!")
        st.rerun()
    if st.session_state.docs or st.session_state.images:
        st.subheader("Processed Files")
        for doc in st.session_state.docs:
            st.markdown(f"📄 {doc['name']}")
        for img in st.session_state.images:
            st.markdown(f"🖼️ {img['name']}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if query := st.chat_input("Ask a question or chat with the AI"):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(query)
            st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})