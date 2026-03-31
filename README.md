
# RAG-Based Document Chatbot

> Upload any PDF or Excel file and ask questions about it — get precise, context-aware answers powered by LLMs and vector search.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-LLM-orange?style=flat-square&logo=google)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red?style=flat-square)

---

## What It Does

Most tools make you read the whole document to find one answer. This chatbot doesn't.

Upload a **PDF or Excel file**, ask a question in plain English, and get a direct answer — extracted from the right part of your document using **Retrieval-Augmented Generation (RAG)**.

---

## How It Works

```
User uploads PDF/Excel
        ↓
Text is extracted & chunked
        ↓
Chunks → Sentence Embeddings (Sentence Transformers)
        ↓
Embeddings stored in Qdrant (Vector DB)
        ↓
User asks a question
        ↓
Question → Embedding → Semantic Search in Qdrant
        ↓
Top matching chunks → sent to Gemini LLM as context
        ↓
LLM generates a precise, grounded answer
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python, Flask |
| LLM | Google Gemini |
| Embeddings | Sentence Transformers |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| Frontend | HTML/CSS |

---

## How to run

### 1. Clone the repo
```bash
git clone https://github.com/3shali/RAG-based-Doc-chatbot.git
cd RAG-based-Doc-chatbot
```

### 2. Create and activate virtual environment
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate
# On Mac/Linux:
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
```bash
# Create a .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 5. Run the app
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## Key Features

- **Multi-format support** — handles both PDF and Excel files
- **Semantic search** — finds relevant content even if you don't use exact keywords
- **Grounded answers** — LLM only answers from your document, reducing hallucinations
- **Simple UI** — no technical knowledge needed to use it

---

## Project Structure

```
RAG-based-Doc-chatbot/
│
├── app.py               # Main Flask app & RAG pipeline
├── templates/           # HTML frontend
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Support for multi-document querying
- [ ] Chat history / memory across turns
- [ ] Deploy on Hugging Face Spaces
- [ ] Add source citation with page numbers

---

## Author

**Thrishali Kotagiri**  
AI & Data Science Graduate | Hyderabad, India  
[LinkedIn](https://linkedin.com/in/thrishali-kotagiri) · [GitHub](https://github.com/3shali)
