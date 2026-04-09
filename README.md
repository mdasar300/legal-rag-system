# ⚖️ AI Techniques for Legal Document Processing using RAG

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 👨‍🎓 Author

* **Name:** Dasari Muralikrishna
* **Student ID:** S2471346
* **Programme:** MSc Computer Science — Glasgow Caledonian University (London)
* **Supervisor:** Dr. Buhong Liu

---

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for legal document processing, semantic understanding, and intelligent information retrieval.

Legal documents are complex, lengthy, and domain-specific. Traditional keyword-based search systems fail to capture semantic meaning, while standalone Large Language Models (LLMs) may generate **hallucinated (incorrect)** responses.

This system addresses these challenges by combining:

* 🔍 **Semantic Retrieval (Embeddings)**
* 🤖 **Grounded Response Generation (LLM)**

The result is a system that produces **accurate, context-aware, and evidence-based answers**.

---

## 🎯 Objectives

* Develop a RAG-based legal information retrieval system
* Improve contextual relevance compared to keyword search
* Reduce hallucination in AI-generated responses
* Support multi-format legal document processing
* Build a practical, end-to-end working prototype

---

## 🧠 System Architecture

```
User Query
    ↓
Query Embedding
    ↓
FAISS Vector Search
    ↓
Relevant Document Chunks
    ↓
LLM (Gemini)
    ↓
Final Answer (Grounded Response)
```

---

## ⚙️ Technologies Used

| Component            | Technology                                 |
| -------------------- | ------------------------------------------ |
| Programming Language | Python 3.11                                |
| User Interface       | Streamlit                                  |
| Embeddings           | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database      | FAISS                                      |
| Language Model       | Gemini API                                 |
| Framework            | LangChain                                  |
| Document Parsing     | PyPDF, python-docx, python-pptx            |
| OCR (Optional)       | Tesseract                                  |

---

## 📂 Supported File Formats

The system supports real-world legal document formats:

* 📄 PDF
* 📝 DOCX
* 📊 PPTX
* 📃 TXT

---

## 🚀 Key Features

* Multi-document upload
* Semantic search (beyond keywords)
* Context-aware retrieval
* Grounded response generation
* Reduced hallucination
* Interactive web interface

---

## 📊 Results Summary

The implementation demonstrates:

* ✔ Improved semantic retrieval over keyword-based methods
* ✔ Contextually relevant and coherent responses
* ✔ Reduced hallucination through grounding
* ✔ Fully functional end-to-end RAG pipeline

---

## ⚠️ Limitations

* Retrieval performance depends on chunking strategy
* Parsing inconsistencies in complex PDFs
* Loss of structure when converting tables/images to text
* No formal quantitative evaluation metrics implemented

---

## 🔮 Future Work

* Implement **RAGAS evaluation framework** (faithfulness, relevance, precision)
* Improve document parsing and structure preservation
* Use domain-specific models (e.g., Legal-BERT)
* Hybrid retrieval (keyword + semantic search)
* Deploy as a scalable production system

---

## ▶️ Installation & Usage

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Application

```bash
streamlit run app.py
```

### 3️⃣ How to Use

1. Upload legal documents
2. Wait for processing and indexing
3. Enter your query
4. View generated response

---

## 📁 Project Structure

```
.
├── app.py
├── requirements.txt
├── README.md
├── data/
├── src/
├── models/
├── utils/
```

---

## 📸 Screenshots (Recommended)

> Add screenshots here before final submission:

* System Interface
* Document Upload
* Retrieval Output
* Generated Response

---

## 🔁 Reproducibility

This project ensures reproducibility through:

* Consistent preprocessing pipeline
* Deterministic embedding model
* Controlled document inputs
* Fixed chunking strategy

---

## 📬 Contact

**Dasari Muralikrishna**
MSc Computer Science — London

---

## ⭐ Acknowledgement

This project is part of an MSc dissertation and demonstrates a practical implementation of Retrieval-Augmented Generation (RAG) for legal AI systems, focusing on improving accuracy, contextual understanding, and trustworthiness in legal information retrieval.

---
