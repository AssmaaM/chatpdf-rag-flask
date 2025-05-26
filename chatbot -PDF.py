
from flask import Flask, request, jsonify
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
import sys
from sqlalchemy import create_engine, text
import fitz  # PyMuPDF
import tempfile
import time
from typing import List
import uuid
import win32api
import win32con

app = Flask(__name__)

# Constants and setup
CHROMA_PATH = "chroma"
DATABASE_URL = "mysql+pymysql://root:@localhost/db"
PORT = 8000  # Specify your desired port number

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def fetch_pdf_data_from_db():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        query = text("SELECT data FROM pdf_files")
        result = connection.execute(query)
        pdf_data_list = [row[0] for row in result]
    return pdf_data_list

def load_documents():
    pdf_data_list = fetch_pdf_data_from_db()
    documents = []

    for pdf_data in pdf_data_list:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name

        try:
            with fitz.open(temp_pdf_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()

                    # Create Document object
                    doc = Document(
                        page_content=page_text,
                        metadata={"source": "database", "page": page_num + 1}
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Error loading PDF: {e}")
        finally:
            os.remove(temp_pdf_path)  # Ensure the temporary file is deleted

    return documents

def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    # Clear existing data in Chroma
    db = None  # Close any existing connection to Chroma
    clear_database()

    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
      
    chunks_with_ids = calculate_chunk_ids(chunks)

    try:
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")
    except Exception as e:
        print(f"Error fetching existing items from Chroma: {e}")
        existing_ids = set()

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        unique_id = str(uuid.uuid4())
        chunk_id = f"{source}:{page}:{unique_id}"
        chunk.metadata["id"] = chunk_id

    return chunks

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    try:
        results = db.similarity_search_with_score(query_text, k=5)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        results = []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        print(f"Error invoking model: {e}")
        response_text = "Error generating response."

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

@app.route('/chatPDF', methods=['POST'])
def chat_pdf():
    user_message = request.json.get('message')
    if user_message is None:
        return jsonify(message="Invalid request: 'message' not found"), 400
    
    # Assuming 'message' is the query for RAG (Retrieve And Generate) response
    response_message = query_rag(user_message)
    return jsonify(message=response_message)

def clear_database():
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError as e:
            print(f"Error clearing database: {e}")
            time.sleep(1)  # Wait a bit and try again
            try:
                win32api.SetFileAttributes(CHROMA_PATH, win32con.FILE_ATTRIBUTE_NORMAL)
                shutil.rmtree(CHROMA_PATH)
            except Exception as e:
                print(f"Retrying error clearing database with win32api: {e}")

def main():
    if "--reset" in sys.argv:
        print("✨ Clearing Database")
        clear_database()

    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents loaded.")
        return

    chunks = split_documents(documents)
    add_to_chroma(chunks)

    # Run Flask app
    app.run(debug=True, port=PORT)

if __name__ == '__main__':
    main()
