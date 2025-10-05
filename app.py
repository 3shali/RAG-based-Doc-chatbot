# app.py

import os
import glob
import sys
import io
import logging
import sqlite3
import numpy as np
import pandas as pd
from uuid import uuid4
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from io import BytesIO

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import easyocr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.docstore.document import Document
import google.generativeai as genai

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load API keys
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

UPLOAD_FOLDER = "data store"
ALLOWED_EXTENSIONS = {"pdf", "xls", "xlsx"}
MODEL = "multi-qa-MiniLM-L6-cos-v1"

embeddings = SentenceTransformer(MODEL)

def embedding_fun(text):
    if isinstance(text, str):
        text = [text]
    return embeddings.encode(text)[0].tolist()

# Qdrant setup
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def clear_data_store(folder_path="data store"):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for file in pdf_files:
        os.remove(file)

def clear_qdrant_collection(collection_name="document_collection"):
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name=collection_name)

def initialize_qdrant_collection(collection_name="document_collection", vectorsize=384):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"content": VectorParams(size=vectorsize, distance=Distance.COSINE)}
        )

def upload_documents_to_qdrant(documents, collection_name="document_collection", batch_size=100):
    chunked_metadata = []
    for item in documents:
        id = str(uuid4())
        content = item.page_content
        source = item.metadata["source"]
        content_vector = embedding_fun(content)
        vector_dict = {"content": content_vector}
        payload = {
            "page_content": content,
            "metadata": {"id": id, "page_content": content, "source": source}
        }
        metadata = PointStruct(id=id, vector=vector_dict, payload=payload)
        chunked_metadata.append(metadata)
        if len(chunked_metadata) >= batch_size:
            qdrant_client.upsert(collection_name=collection_name, wait=True, points=chunked_metadata)
            chunked_metadata = []
    if chunked_metadata:
        qdrant_client.upsert(collection_name=collection_name, wait=True, points=chunked_metadata)

ocr_reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf(pdf_path):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    row_text = "\t".join([cell if cell else "" for cell in row])
                    full_text.append(row_text)
    return "\n".join(full_text)

def extract_text_from_images(pdf_path):
    text_blocks = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = images[img_index][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = ocr_reader.readtext(np.array(image), detail=0)
            text_blocks.append("\n".join(result))
    return "\n".join(text_blocks)

def load_documents_from_folder(data_folder="data store"):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            try:
                text = extract_text_from_pdf(pdf_path)
                img_text = extract_text_from_images(pdf_path)
                combined_text = text + "\n\n[Image Texts]\n" + img_text if img_text else text
                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                chunks = splitter.split_text(combined_text)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"source": filename}))
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
    return documents

# Gemini LLM
class GeminiLLM:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

def strip_code_fence(text):
    if text.strip().startswith("```") and text.strip().endswith("```"):
        return "\n".join(text.strip().split("\n")[1:-1]).strip()
    return text.strip()

# Excel processor
class ExcelProcessor:
    def __init__(self, db_path="excel_data.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def get_prompt(self, sample_df, table_name):
        return f"""
You are an expert database assistant.
Given this table:

{sample_df.to_csv(index=False)}

Generate:
- MySQL CREATE TABLE {table_name} with correct column types (INT, FLOAT, DATE, VARCHAR)
- Add a PRIMARY KEY if logical
- INSERT INTO {table_name} statements for the rows.

Only return valid MySQL SQL code.
"""

    def execute_sql(self, sql_code):
        try:
            sql_code = strip_code_fence(sql_code)
            logging.info(f"[CREATE/INSERT SQL]:\n{sql_code}")
            self.cursor.executescript(sql_code)
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"SQL execution failed: {e}")

    def process_excel_file(self, file_content, table_name="uploaded_table"):
        df = pd.read_excel(BytesIO(file_content))

    # Generate CREATE TABLE only from first 5 rows
        create_prompt = self.get_prompt(df, table_name)
        sql_code = llm.generate_response(create_prompt)
        self.execute_sql(sql_code)

    # Now insert all rows using pandas directly
        try:
            df.to_sql(table_name, self.conn, if_exists="append", index=False)
            logging.info(f"Inserted {len(df)} rows into table {table_name}")
        except Exception as e:
            logging.error(f"Pandas insert failed: {e}")

    def query_data(self, question, table_name="uploaded_table"):
        self.cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [row[1] for row in self.cursor.fetchall()]
        column_list = ", ".join(columns)

    # Step 1: Generate SQL
        prompt = f"""
    You are a helpful assistant who converts natural language to SQL.
    The table "{table_name}" has the following columns:
    {column_list}

    User question: "{question}"
    Only return a valid SQL SELECT query using **only the columns above**, and nothing else.
    """
        sql_query = llm.generate_response(prompt)
        sql_query = strip_code_fence(sql_query)
        logging.info(f"[SELECT SQL for question '{question}']:\n{sql_query}")

        try:
            result = self.cursor.execute(sql_query).fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(result, columns=columns)

        # Step 2: Use Gemini to format the result into natural language
            result_text = df.to_string(index=False)
            explanation_prompt = f"""
    You are a data analyst assistant.
    Given the user's question and this table result:

    Question: "{question}"
    SQL Result:
    {result_text}


    Provide a clear and concise natural language answer.
    """
            return llm.generate_response(explanation_prompt)

        except Exception as e:
            logging.error(f"Query failed: {e}")
            return "Sorry, could not execute your query."

        
retriever = Qdrant(client=qdrant_client, collection_name="document_collection", embeddings=embedding_fun, vector_name="content").as_retriever(search_kwargs={"k": 2})
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = GeminiLLM()
excel_processor = ExcelProcessor()

def build_prompt(context, question, history):
    history_text = "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage) else f"Bot: {m.content}" for m in history])
    return f"""
{history_text}
Context: {context}

Question: {question}
Answer:
"""

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    answer = question = context = None
    history = chat_memory.load_memory_variables({})["chat_history"]

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            ext = file.filename.rsplit(".", 1)[-1].lower()
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if ext == "pdf":
                chunks = load_documents_from_folder()
                upload_documents_to_qdrant(chunks)
            elif ext in ("xls", "xlsx"):
                file.seek(0)
                excel_processor.process_excel_file(file.read())

        elif "question" in request.form:
            question = request.form["question"]
            docs = retriever.get_relevant_documents(question)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                prompt = build_prompt(context, question, history)
                answer = llm.generate_response(prompt)
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
                source_str = ", ".join(os.path.basename(src) for src in sources)
            else:
                answer = excel_processor.query_data(question)
                source_str = "Excel"

            chat_memory.chat_memory.messages.append(HumanMessage(content=f"{question} (📄 {source_str})"))
            chat_memory.chat_memory.messages.append(AIMessage(content=f"{answer} (📄 {source_str})"))

    history = chat_memory.load_memory_variables({})["chat_history"]
    return render_template("index.html", question=question, context=context, answer=answer, chat_history=history)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    clear_data_store()
    clear_qdrant_collection()
    initialize_qdrant_collection()
    app.run(debug=True)
