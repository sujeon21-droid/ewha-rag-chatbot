import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Ewha Admission RAG Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 Ewha Admission RAG Chatbot")
st.caption("Ask questions in English about the uploaded Ewha admission guide. Answers are generated only from retrieved document chunks.")

DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-R1"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PDF_PATH = "data/ewha.pdf"


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    keywords = [
        "application",
        "schedule",
        "announcement",
        "documents",
        "admission",
        "result",
        "scholarship",
        "international student affairs",
        "online application",
        "deadline",
        "requirement",
    ]

    filtered_docs = []
    for doc in split_docs:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in keywords):
            filtered_docs.append(doc)

    if not filtered_docs:
        filtered_docs = split_docs

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(filtered_docs, embeddings)
    return vectorstore, len(docs), len(split_docs), len(filtered_docs), file_name


def build_llm(api_key: str, model_name: str, base_url: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.6,
    )


def answer_question(llm: ChatOpenAI, retriever, question: str):
    related_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in related_docs)
    prompt = f"""
You are an admissions assistant for Ewha Womans University.
Answer the user's question only based on the retrieved document excerpts below.
Use English only.
If the answer is not explicitly supported by the excerpts, say: "I could not find that information in the uploaded document. Please verify it on the official university website."
Be concise and factual.

Retrieved document excerpts:
{context}

User question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content, related_docs


with st.sidebar:
    st.header("Settings")
    default_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    api_key = st.text_input("SiliconFlow API Key", value=default_api_key, type="password")
    model_name = st.text_input("Model name", value=DEFAULT_MODEL)
    base_url = DEFAULT_BASE_URL
    top_k = st.slider("Number of retrieved chunks", min_value=3, max_value=10, value=6)
    st.markdown("Recommended model: `Pro/deepseek-ai/DeepSeek-R1`")
    st.markdown("Base URL is fixed to `.cn/v1` for this app.")

if not os.path.exists(DEFAULT_PDF_PATH):
    st.error(f"Default PDF not found: {DEFAULT_PDF_PATH}")
    st.stop()

try:
    with open(DEFAULT_PDF_PATH, "rb") as f:
        file_bytes = f.read()

    vectorstore, page_count, chunk_count, filtered_count, stored_name = build_vectorstore(
        file_bytes, os.path.basename(DEFAULT_PDF_PATH)
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", page_count)
    col2.metric("Chunks", chunk_count)
    col3.metric("Indexed chunks", filtered_count)

    st.success(f"Document indexed successfully: {stored_name}")
except Exception as exc:
    st.error(f"Failed to process the PDF: {exc}")
    st.stop()

question = st.text_input(
    "Ask a question in English",
    value="When is the online application period for Fall 2026?",
)

if st.button("Get answer"):
    if not api_key:
        st.error("Please enter your SiliconFlow API key in the sidebar.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            llm = build_llm(
                api_key=api_key.strip(),
                model_name=model_name.strip(),
                base_url=base_url,
            )
            with st.spinner("Generating answer..."):
                answer, sources = answer_question(llm, retriever, question.strip())

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved sources")
            for idx, doc in enumerate(sources, start=1):
                page_num = doc.metadata.get("page")
                page_label = f"Page {page_num + 1}" if isinstance(page_num, int) else "Unknown page"
                with st.expander(f"Source {idx} — {page_label}"):
                    st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
