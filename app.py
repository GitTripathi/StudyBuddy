"""StudyBuddy: upload a PDF, ask questions. A small AI agent with 2 tools.

Run:  streamlit run app.py
"""
import ast
import operator as op
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load API key from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Load Gemini model from Streamlit Secrets
MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
)
SYSTEM = (
    "You are StudyBuddy, a study assistant. For questions about the user's notes, "
    "call search_notes and answer ONLY from what it returns. If the notes do not "
    "contain the answer, say so. Use calculator for any maths."
)

# ---------- Step 1: RAG (turn the PDF into a searchable index) ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        temp_path = f.name

    try:
        pages = PyPDFLoader(temp_path).load()

        # Remove empty pages
        pages = [
            page for page in pages
            if page.page_content and page.page_content.strip()
        ]

        if not pages:
            raise ValueError(
                "No readable text found in PDF. "
                "Please upload a text-based PDF."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(pages)

        if not chunks:
            raise ValueError("No text chunks generated from PDF.")

        return FAISS.from_documents(
            chunks,
            get_embeddings()
        )

    finally:
        os.remove(temp_path)
# ---------- Step 2: calculator (safe, no eval) ----------
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}


def safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    raise ValueError("unsupported")


# ---------- Step 3: the agent loop ----------
def run_agent(question, store):
    @tool
    def search_notes(query: str) -> str:
        """Search the uploaded notes for text relevant to the query."""
        return "\n\n".join(d.page_content for d in store.similarity_search(query, k=3))

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a maths expression such as '15/100*2340'."""
        try:
            return str(safe_eval(ast.parse(expression, mode="eval").body))
        except Exception:
            return "Invalid expression"

    tools = {t.name: t for t in (search_notes, calculator)}
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0).bind_tools(list(tools.values()))
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    used = []

    for _ in range(5):                                       # think -> act -> observe
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:                             # no tool needed: final answer
            text = reply.content
            if isinstance(text, list):
                text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
            return text, used
        for call in reply.tool_calls:                        # LLM chose a tool: run it
            result = tools[call["name"]].invoke(call["args"])
            used.append(call["name"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "Sorry, I could not finish that.", used


# ---------- Step 4: Streamlit UI ----------
st.title("📚 StudyBuddy")
st.caption("Upload your notes (PDF) and ask questions.")

pdf = st.file_uploader("Upload notes", type="pdf")
if pdf and st.session_state.get("name") != pdf.name:
    with st.spinner("Reading your notes..."):
        st.session_state.store = build_index(pdf.getvalue())
        st.session_state.name = pdf.name
        st.session_state.chat = []

if "store" not in st.session_state:
    st.info("Upload a PDF to start.")
    st.stop()

for role, text in st.session_state.chat:
    st.chat_message(role).write(text)

if question := st.chat_input("Ask about your notes..."):
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        answer, used = run_agent(question, st.session_state.store)
    if used:
        answer += f"\n\n*Tools used: {', '.join(used)}*"
    st.chat_message("assistant").write(answer)
    st.session_state.chat += [("user", question), ("assistant", answer)]
