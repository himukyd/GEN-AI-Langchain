# 🚀 GEN-AI-LangChain

A complete hands-on journey into **Generative AI using LangChain** covering:

- LLMs
- Prompt Engineering
- Embeddings
- Vector Similarity
- RAG (Retrieval-Augmented Generation)
- Chat Models
- Hugging Face
- Google Gemini
- OpenAI
- Document Retrieval
- Semantic Search
- Agents
- Memory
- LangGraph *(upcoming)*

Built while learning and experimenting with modern GenAI workflows.

---

# 📚 About This Repository

This repository contains my practical learning and implementation of **Generative AI using LangChain**.

The goal of this repository is to:
- Learn GenAI from basics to advanced
- Understand LangChain deeply
- Work with real-world LLM pipelines
- Build RAG applications
- Explore embeddings and vector databases
- Experiment with multiple AI providers

---

# 🛠️ Technologies Used

## 🧠 LLM Providers

- OpenAI
- Google Gemini
- Hugging Face

---

# ⚙️ Frameworks & Libraries

- LangChain
- LangGraph
- Transformers
- Sentence Transformers
- Scikit-learn
- NumPy
- Python
- dotenv

---

# 📂 Project Structure

```bash
GEN-AI-LangChain/
│
├── 1.Prompt_Template/
├── 2.Chat_Model/
├── 3.Embedded_Model/
├── 4.Output_Parser/
├── 5.Chains/
├── 6.Document_Similarity/
├── 7.RAG/
├── 8.Vector_Database/
├── 9.Memory/
├── 10.Agents/
├── 11.LangGraph/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🧠 Topics Covered

## ✅ Prompt Engineering

- Prompt Templates
- Dynamic Prompts
- Chat Prompts
- System & Human Messages

---

# 🤖 Chat Models

Working with:
- OpenAI Chat Models
- Gemini Models
- Hugging Face Chat Models

Example:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro"
)
```

---

# 🔍 Embeddings

Implemented:
- Query Embeddings
- Document Embeddings
- Similarity Search
- Semantic Retrieval

Embedding Providers:
- Google Gemini Embeddings
- Hugging Face Embeddings
- OpenAI Embeddings

---

# 📄 Document Similarity

Implemented cosine similarity using embeddings.

Example workflow:

```text
Documents
↓
Embeddings
↓
Cosine Similarity
↓
Most Relevant Document
```

---

# 📚 RAG (Retrieval-Augmented Generation)

Learning and building:
- Document Retrieval
- Chunking
- Embedding Pipelines
- Retriever Systems
- Context Injection
- QA Systems

---

# 🧮 Vector Databases

Exploring:
- FAISS
- ChromaDB
- Pinecone *(upcoming)*

---

# 🧠 Memory in LangChain

Working with:
- Conversation Buffer Memory
- Conversation Summary Memory
- Chat History

---

# 🤖 Agents

Learning:
- Tool Calling
- Multi-step Reasoning
- Autonomous Workflows

---

# 🔄 LangGraph *(Upcoming)*

Planned topics:
- Stateful AI Workflows
- Graph-based Agents
- Multi-agent Systems

---

# 🚀 Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/GEN-AI-LangChain.git
```

---

# 2️⃣ Create Virtual Environment

```bash
python -m venv him_lanchain
```

Activate:

## Mac/Linux

```bash
source him_lanchain/bin/activate
```

## Windows

```bash
him_lanchain\Scripts\activate
```

---

# 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Create `.env`

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

# ▶️ Run Example

```bash
python3 embedding_google_query.py
```

---

# 📌 Example: Document Similarity

```python
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)
```

---

# 🎯 Learning Goals

- Understand how LLMs work
- Learn LangChain architecture
- Build production-ready RAG systems
- Explore vector databases
- Understand embeddings deeply
- Work with multiple AI providers
- Build AI agents and workflows

---

# 📈 Future Plans

- PDF Question Answering
- AI Chatbot
- Multi-Agent Systems
- LangGraph Workflows
- AI Research Assistant
- Local LLM Integration
- Fine-Tuning
- AI Voice Assistant

---

# 🤝 Contributions

Contributions, suggestions, and improvements are welcome.

Feel free to:
- Fork the repository
- Create pull requests
- Open issues

---

# 📬 Connect With Me

## 👨‍💻 Himanshu

- Master's in Mathematics & Computing, IIT Bhilai
- Interested in:
  - Generative AI
  - NLP
  - Machine Learning
  - RAG Systems
  - AI Agents

GitHub:  
https://github.com/himukyd

---

# ⭐ If You Like This Repository

Give it a ⭐ on GitHub and follow my GenAI learning journey.
