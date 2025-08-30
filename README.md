# 🩺 Medical LLM Assistant

An **AI-powered assistant** built with **LangChain, HuggingFace embeddings, FAISS, and OpenRouter LLMs**.  
This project allows you to **chat with medical research papers, clinical trial reports, or notes** (PDF, TXT, DOCX).  
It helps researchers, students, and healthcare professionals quickly find relevant insights from scientific documents.

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard Overview](https://github.com/pandeyshikhar18/medical-llm-assistant/blob/main/data/dashboard.png)

### Evaluation Results
![Evaluation Results](https://github.com/pandeyshikhar18/medical-llm-assistant/blob/main/data/eval.png)

---

## 🚀 Features
- 📂 **Multi-format support**: Upload **PDF, TXT, DOCX** files.  
- 🔍 **RAG (Retrieval-Augmented Generation)**: Extracts chunks, embeds them using **sentence-transformers**, and retrieves context with **FAISS**.  
- 🤖 **LLM-powered Q&A**: Uses **OpenRouter-hosted LLMs** (Mistral, LLaMA, etc.) for contextual answers.  
- 📝 **Citations**: Answers always include **document name + page/line reference**.  
- 📊 **Evaluation Module**: Includes `evaluate.py` to compute **Recall@k** and **NDCG@k** for retrieval performance.  
- 🎨 **Streamlit UI**: Clean chat interface with custom CSS for a professional look.  
- 🐳 **Dockerized Deployment**: Easily containerized with Docker for reproducibility.  

---

## 📂 Project Structure
medical-llm-assistant/
│── app.py # Main Streamlit app
│── evaluate.py # Evaluation script for retrieval quality
│── requirements.txt # Python dependencies
│── htmlTemplates.py # Chat UI templates (user/bot)
│── .env # API keys (kept private, not pushed to GitHub)
│── data/
│ ├── vaccines.pdf
│ ├── covid19.pdf
│ └── ...
│── Dockerfile # Docker build instructions
│── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Tech Stack
- **Python 3.11**
- **Streamlit** → UI framework  
- **LangChain** → Conversational Retrieval Chain  
- **HuggingFace Sentence Transformers** → Embeddings (`all-MiniLM-L6-v2`)  
- **FAISS** → Vector search for RAG  
- **OpenRouter LLMs** → Chat-based reasoning (Mistral-7B-Instruct used by default)  
- **Docker** → Containerization  

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/pandeyshikhar18/medical-llm-assistant.git
cd medical-llm-assistant
2️⃣ Create virtual environment & install dependencies
bash
Copy code
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac

pip install -r requirements.txt
3️⃣ Add API Key
Create a .env file in the project root:

ini
Copy code
OPENROUTER_API_KEY=your_openrouter_api_key_here
4️⃣ Run Streamlit app
bash
Copy code
streamlit run app.py
Open in browser → http://localhost:8501

🧪 Evaluation (Retrieval Quality)
We provide an evaluation script evaluate.py that calculates Recall@k and NDCG@k for retrieval.

Run:

bash
Copy code
python evaluate.py
Example Output:

kotlin
Copy code
📊 Evaluation Results
Recall@10: 0.50
NDCG@10:   0.50
{'recall_at_k': 0.5, 'ndcg_at_k': 0.5, 'k': 10, 'queries_evaluated': 2}
🐳 Docker Deployment
1️⃣ Build Docker Image
bash
Copy code
docker build -t medical-llm-assistant .
2️⃣ Run Container with .env
bash
Copy code
docker run -d -p 8501:8501 --env-file .env medical-llm-assistant
3️⃣ Access the App
Open → http://localhost:8501

🎯 Example Use Cases
Summarize findings from COVID-19 treatment trials

Extract side effects of a drug from research papers

Compare methodologies across oncology studies

Quickly reference clinical trial outcomes

📌 Future Improvements
Add multi-modal support (medical images + text).

Integrate rerankers (cross-encoders) for better retrieval.

Expand evaluation to include precision, F1-score, MRR.

Add user authentication for secure access.

👨‍💻 Author
Shikhar Pandey
🔗 GitHub: pandeyshikhar18


