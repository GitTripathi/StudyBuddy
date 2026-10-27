# 📚 StudyBuddy: AI Document Q&A Chatbot

StudyBuddy lets you upload a PDF of your notes and ask questions about it in plain English. It is a small **AI agent**: the LLM decides on its own whether to search your notes or use a calculator, then answers using only what it found.

Built with Python, LangChain, FAISS and Streamlit.

---

## ✨ Features
- 📄 Upload any text-based PDF (notes, textbook chapter, research paper)
- 🔍 Answers come from your document using **RAG** (Retrieval-Augmented Generation), so it doesn't make things up
- 🤖 **AI agent with 2 tools:** `search_notes` and `calculator`. The LLM chooses which one to use
- 💬 Simple chat interface built with Streamlit
- 🆓 Free to run: local embeddings + Gemini free tier

## 🛠️ Tech Stack
| Part | Technology |
|------|-----------|
| Language | Python |
| LLM framework | LangChain |
| LLM | Google Gemini API |
| Vector database | FAISS |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Frontend | Streamlit |
| Version control | Git, GitHub |

## 🧠 How It Works
```
        ┌──────────── Indexing (once per PDF) ────────────┐
PDF ──► split into chunks ──► embeddings ──► FAISS index

        ┌──────────── Answering (every question) ─────────┐
Question ──► LLM (agent) ──► picks a tool
                              ├─ search_notes ──► top 3 matching chunks ─┐
                              └─ calculator   ──► result ────────────────┤
                                                                         ▼
                                                     LLM writes the final answer
```

1. **Chunking:** the PDF is split into small overlapping pieces.
2. **Embeddings:** each chunk is converted into a list of numbers that captures its meaning, then stored in FAISS.
3. **Agent loop:** the LLM reads the question and decides which tool to call (think → act → observe).
4. **Retrieval:** `search_notes` finds the chunks most similar to the question.
5. **Answer:** the LLM answers using only those chunks. If the notes don't contain the answer, it says so.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or newer
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation
```bash
git clone https://github.com/GitTripathi/studybuddy.git
cd studybuddy

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env            # Windows: copy .env.example .env
```
Open `.env` and paste your key:
```
GOOGLE_API_KEY=your_key_here
```

### Run
```bash
streamlit run app.py
```
Then open http://localhost:8501, upload a PDF and start asking questions.

> The first run downloads the embedding model (about 90 MB), so it may take a minute.

## 💡 Example Questions
- "Summarise what my notes say about normalization"
- "What is the difference between a process and a thread?"
- "What is 15% of 2340?"  *(uses the calculator tool)*

## 📁 Project Structure
```
studybuddy/
├── app.py              # RAG pipeline, agent loop, tools and Streamlit UI
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key
├── .gitignore
└── README.md
```

## ⚠️ Limitations
- Works only with text-based PDFs (scanned images need OCR)
- The index is kept in memory, so it resets when the app restarts
- Handles one document at a time

## 🔮 Future Improvements
- [ ] Add more tools (web search, quiz generator)
- [ ] Chat memory for follow-up questions
- [ ] Support multiple documents
- [ ] Save the FAISS index to disk
- [ ] Deploy on Streamlit Cloud

## 📸 Screenshot
_Add a screenshot or demo GIF of the app here._

## 👤 Author
**Aryan Tripathi**, B.Tech EIE, NIT Silchar
GitHub: [GitTripathi](https://github.com/GitTripathi)
