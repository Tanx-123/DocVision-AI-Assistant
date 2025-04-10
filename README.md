# DocVision AI Assistant

> Your Intelligent Document and Image Analysis Companion

An advanced chatbot that combines document processing with image analysis capabilities, powered by Streamlit and Ollama's large language models. Perfect for analyzing documents, extracting information from images, and providing context-aware responses.

## Features

- 📄 Multi-format document processing (PDF, DOCX, TXT, CSV, XLSX)
- 🖼️ Image analysis with text extraction
- 💬 Context-aware responses using TF-IDF vectorization
- 🤖 Chat interface with conversation history
- 📱 Responsive web interface built with Streamlit

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Requirements

- Python 3.8+
- Ollama with models:
  - gemma3:12b
  - llama3.2-vision:latest

## Usage

1. **Upload Files**
   - Click "Upload Files" in sidebar
   - Support for images (PNG, JPG, JPEG) and documents

2. **Ask Questions**
   - Type questions about uploaded files
   - Use natural language for queries
   - Reference images or documents directly

3. **View Results**
   - See processed files in sidebar
   - Get AI-generated responses
   - View extracted text and analysis

## Technical Details

- **Vectorization**: TF-IDF for document similarity
- **Chunking**: Text split into 500-word chunks with 50-word overlap
- **Image Analysis**: Uses llama3.2-vision model
- **Text Generation**: Uses gemma3:12b model

## License
MIT License - See LICENSE file