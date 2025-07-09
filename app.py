import streamlit as st
import os
import re
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer
import shutil

# --- Konfigurasi ---
PDF_FOLDER = "pdf_files"
DB_FOLDER = "./rag_db"
COLLECTION_NAME = "student_handbooks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_CONTEXT_LIMIT = 4096
RESPONSE_TOKENS = 128
BUFFER_TOKENS = MODEL_CONTEXT_LIMIT - RESPONSE_TOKENS

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Fungsi PDF 
def test_pdf_open(file_path):
    try:
        doc = fitz.open(file_path)
        print(f"✅ Berhasil membuka PDF: {file_path}")
        return True
    except Exception as e:
        print(f"⚠ Gagal membuka PDF: {e}")
        return False

def extract_text_with_fitz(file_path):
    text = ""
    if test_pdf_open(file_path):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def split_text_into_chunks(text, max_tokens=512):
    words = text.split()
    chunks = []
    while words:
        chunk = " ".join(words[:max_tokens])
        chunks.append(chunk)
        words = words[max_tokens:]
    return chunks

def load_documents_from_pdfs():
    documents = []
    ids = []
    doc_id = 0
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            if test_pdf_open(file_path):
                text = extract_text_with_fitz(file_path)
                cleaned_text = clean_text(text)
                if cleaned_text and len(cleaned_text.split()) > 5:
                    for chunk in split_text_into_chunks(cleaned_text):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={"filename": filename},
                            id=str(doc_id)
                        ))
                        ids.append(str(doc_id))
                        doc_id += 1
    return documents, ids

# Inisialisasi ChromaDB 
if os.path.exists(DB_FOLDER):
    try:
        shutil.rmtree(DB_FOLDER)
        print("🔄 Database dihapus untuk mencegah lock error.")
    except Exception as e:
        print(f"⚠ Gagal menghapus database: {e}")

add_documents = not os.path.exists(DB_FOLDER)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_FOLDER,
    embedding_function=embeddings
)
if add_documents:
    docs, ids = load_documents_from_pdfs()
    print(f"📂 Total dokumen yang dimasukkan ke ChromaDB: {len(docs)}")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# LLM 
llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

#  RAG Query 
def rag_query(query):
    results = retriever.invoke(query)
    print(f"🔎 {len(results)} dokumen ditemukan")

    base_prompt = "Jawablah pertanyaan berikut berdasarkan konteks yang diberikan.\n\nKonteks:\n"
    query_part = f"\n\nPertanyaan: {query}\nJawaban:"
    fixed_tokens = count_tokens(base_prompt + query_part)
    max_tokens_for_context = MODEL_CONTEXT_LIMIT - RESPONSE_TOKENS - fixed_tokens

    context_chunks = []
    used_sources = []
    total_context_tokens = 0
    for doc in results:
        chunk = doc.page_content
        tokens = count_tokens(chunk)
        if total_context_tokens + tokens > max_tokens_for_context:
            continue
        context_chunks.append(f"[{doc.metadata.get('filename', 'Tidak ada metadata')}]\n{chunk}")
        used_sources.append(doc.metadata.get("filename", "Tidak ada metadata"))
        total_context_tokens += tokens

    if not context_chunks and results:
        doc = min(results, key=lambda d: count_tokens(d.page_content))
        context_chunks.append(f"[{doc.metadata.get('filename', 'Tidak ada metadata')}]\n{doc.page_content}")
        used_sources.append(doc.metadata.get("filename", "Tidak ada metadata"))

    context = "\n\n".join(context_chunks)
    full_prompt = f"{base_prompt}{context}{query_part}"
    response = llm.stream(full_prompt)
    answer = "".join(chunk for chunk in response)

    return answer.strip(), used_sources

# STREAMLIT UI 
st.set_page_config(page_title="TanyaVara - Handbook Assistant", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        /* Mengatur container utama */
        .main, .block-container {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            text-align: left;
            padding-top: 20px;
            padding-bottom: 20px;
            width: 75%;
        }

        /* Chat container style */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            width: 90%;
            margin: auto;
        }

        /* Gaya untuk pesan pengguna */
        .user-message {
            padding: 14px 18px;
            border-radius: 18px;
            margin-bottom: 12px;
            max-width: 75%;
            align-self: flex-end;
            font-size: 20px;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        /* Gaya untuk pesan bot */
        .bot-message {
            padding: 14px 18px;
            border-radius: 18px;
            margin-bottom: 12px;
            max-width: 75%;
            align-self: flex-start;
            font-size: 20px;
            word-wrap: break-word;
            border: 1px solid #ddd;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        /* Sumber dokumen */
        .source-box {
            background-color: #f0f2f6;
            padding: 0.75em;
            border-left: 5px solid #004d7a;
            margin-top: 0.5em;
            font-size: 16px;
            margin-left: 25px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        /* Styling untuk input form chat */
        .stTextInput>div>div>input {
            background-color: #333;
            color: #fff;
            font-size: 24px;  
            padding: 30px 35px; 
            border-radius: 15px; 
            width: 500px;  
            max-width: 90%;  
            margin: auto;
            box-sizing: border-box; 
        }

        .stButton>button {
            background-color: #004d7a;
            color: white;
            font-size: 16px;
            padding: 12px;
            border-radius: 12px;
            width: 100%;
        }

        .stButton>button:hover {
            background-color: #0066a1;
        }

        /* Menambahkan padding untuk responsif di perangkat mobile */
        @media (max-width: 768px) {
            .chat-container {
                width: 100%;
            }

            .user-message, .bot-message {
                font-size: 18px;
            }

            .stTextInput>div>div>input {
                font-size: 20px;  /* Ukuran font lebih besar di mobile */
                width: 80%;  /* Membatasi lebar di mobile */
            }

            .stButton>button {
                font-size: 18px;
            }
        }

        /* Penyesuaian warna untuk mode terang dan gelap */
        /* Mode Terang */
        @media (prefers-color-scheme: light) {
            .user-message {
                background-color: #DCF8C6;
                color: #333; /* Teks gelap agar kontras di latar terang */
            }
            .bot-message {
                background-color: #FFFFFF;
                color: #333; /* Teks gelap agar kontras di latar terang */
            }
            .stTextInput>div>div>input {
                background-color: #fff;
                color: #333;
            }
            .stButton>button {
                background-color: #004d7a;
                color: white;
            }
            .source-box {
                background-color: #f0f2f6;
                color: #333;
            }
        }

        /* Mode Gelap */
        @media (prefers-color-scheme: dark) {
            .user-message {
                background-color: #4A6E33; /* Warna lebih gelap untuk user di mode gelap */
                color: #fff; /* Teks terang untuk kontras */
            }
            .bot-message {
                background-color: #333333;
                color: #fff; /* Teks terang agar kontras di latar gelap */
            }
            .stTextInput>div>div>input {
                background-color: #444;
                color: #fff;
            }
            .stButton>button {
                background-color: #004d7a;
                color: white;
            }
            .source-box {
                background-color: #333;
                color: #fff;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style="font-size: 45px; font-family: 'Arial', sans-serif;">📘 TanyaVara - Asisten Handbook Mahasiswa</h1>
    <p style="font-size: 20px; font-family: 'Arial', sans-serif;">💬 Selamat datang! Tanya apapun seputar buku panduan mahasiswa. Chatbot akan membantu memberikan jawaban yang dirangkum dari dokumen handbook kampus.</p>
""", unsafe_allow_html=True)


# Inisialisasi riwayat chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input chat dari user
user_input = st.chat_input("Ketik pertanyaan Anda di sini...")

# Jika ada pertanyaan baru
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.spinner("Sedang mencari jawaban..."):
        answer, sources = rag_query(user_input)
    st.session_state.chat_history.append({"role": "bot", "text": answer, "sources": sources})

# Tampilkan riwayat chat
st.markdown("### 🗨️ Obrolan:")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='chat-container'><div class='user-message'>🙋‍♂️ {message['text']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-container'><div class='bot-message'>🤖 {message['text']}</div></div>", unsafe_allow_html=True)
        if message.get("sources"):
            for src in set(message["sources"]):
                st.markdown(f"<div class='source-box'>📎 {src}</div>", unsafe_allow_html=True)



