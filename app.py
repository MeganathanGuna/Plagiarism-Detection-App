import os
import ctypes

# ────────────────────────────────────────────────
# CRITICAL: Pre-load c10.dll BEFORE ANY OTHER IMPORTS
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch_lib_dir = os.path.join(script_dir, 'venv', 'Lib', 'site-packages', 'torch', 'lib')
    c10_full_path = os.path.join(torch_lib_dir, 'c10.dll')

    if os.path.exists(c10_full_path):
        ctypes.CDLL(c10_full_path)
        print("Successfully pre-loaded c10.dll")
    else:
        print(f"WARNING: c10.dll not found at: {c10_full_path}")
except Exception as e:
    print(f"c10.dll pre-load failed: {e}")

# ────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title='TrueSource — Text Origin & Similarity',
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

import pandas as pd
import nltk
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load semantic model
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()

# ────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────

def get_sentences(text):
    if not text or not text.strip():
        return []
    return tokenize.sent_tokenize(text.strip())

def google_search_url(sentence):
    if not sentence or not sentence.strip():
        return None

    api_key = "4cca66d7cd1dc966c98ffbdf6c026fdb96dbe0e5"   # Your Serper key

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": sentence.strip()})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'organic' in data and data['organic']:
            return data['organic'][0]['link']
        elif 'answerBox' in data and 'link' in data['answerBox']:
            return data['answerBox']['link']
    except Exception as e:
        print(f"Serper error: {e}")
    return None

def extract_text_from_url(url):
    if not url:
        return ""
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        return text[:4000]
    except:
        return ""

def get_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    t = uploaded_file.type
    if t == "text/plain":
        with io.TextIOWrapper(uploaded_file, encoding='utf-8') as f:
            return f.read()
    elif t == "application/pdf":
        text = ""
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + " "
        return text
    elif t == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    return ""

def lexical_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        cv = CountVectorizer(stop_words='english')
        matrix = cv.fit_transform([text1, text2])
        return cosine_similarity(matrix)[0][1]
    except:
        return 0.0

def semantic_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = semantic_model.encode(text1, convert_to_tensor=False)
    emb2 = semantic_model.encode(text2, convert_to_tensor=False)
    return util.cos_sim(emb1, emb2).item()

# ────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────

st.title("TrueSource — Text Origin & Similarity Detection")
st.markdown("Detect plagiarism, paraphrasing, and find original sources using **semantic embeddings** + web search.")

option = st.radio(
    "Select mode:",
    ('Paste text', 'Upload single file', 'Compare multiple files'),
    horizontal=True
)

if option == 'Paste text':
    text_input = st.text_area("Paste your text here", height=220, key="paste_text")
elif option == 'Upload single file':
    uploaded_file = st.file_uploader("Upload .txt / .pdf / .docx", type=["txt", "pdf", "docx"], key="single_file")
    text_input = ""
    if uploaded_file:
        with st.spinner("Reading file..."):
            text_input = get_text_from_file(uploaded_file)
        st.success("File loaded!")
else:
    uploaded_files = st.file_uploader(
        "Upload multiple files (.txt/.pdf/.docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="multi_files"
    )
    if uploaded_files:
        texts = []
        filenames = []
        with st.spinner("Reading files..."):
            for uf in uploaded_files:
                content = get_text_from_file(uf)
                if content.strip():
                    texts.append(content)
                    filenames.append(uf.name)
        if texts:
            st.session_state['multi_texts'] = texts
            st.session_state['multi_filenames'] = filenames
            st.success(f"{len(texts)} files loaded!")

if st.button("Analyze / Check for Plagiarism", type="primary"):
    if option in ('Paste text', 'Upload single file'):
        text = text_input.strip()
        if not text:
            st.error("No text provided.")
            st.stop()

        with st.spinner("Analyzing sentences & searching web..."):
            sentences = get_sentences(text)
            if not sentences:
                st.warning("No sentences found.")
                st.stop()

            results = []
            progress_bar = st.progress(0)
            for idx, sent in enumerate(sentences):
                url = google_search_url(sent)
                source_text = extract_text_from_url(url) if url else ""
                lex_sim = lexical_similarity(sent, source_text)
                sem_sim = semantic_similarity(sent, source_text)

                results.append({
                    'Sentence': sent,
                    'Source URL': url or "",
                    'Lexical Similarity': round(lex_sim * 100, 1),
                    'Semantic Similarity (%)': round(sem_sim * 100, 1)
                })
                progress_bar.progress((idx + 1) / len(sentences))

            df = pd.DataFrame(results)
            df = df.sort_values('Semantic Similarity (%)', ascending=False).reset_index(drop=True)

            # Lowered thresholds for better risk detection
            high_matches = df[df['Semantic Similarity (%)'] > 50]
            risk = round(high_matches['Semantic Similarity (%)'].mean(), 1) if not high_matches.empty else 0.0
            st.subheader(f"Overall Plagiarism Risk: **{risk}%** (semantic)")

            def highlight_row(row):
                if row['Semantic Similarity (%)'] >= 60:
                    return ['background-color: #ffcccc'] * len(row)
                elif row['Semantic Similarity (%)'] >= 45:
                    return ['background-color: #fff3cd'] * len(row)
                return [''] * len(row)

            styled = df.style.apply(highlight_row, axis=1)
            st.markdown("**Results** (sorted by semantic similarity — red = high risk)")
            st.dataframe(
                styled,
                use_container_width=True,
                column_config={"Source URL": st.column_config.LinkColumn("Source URL")}
            )

    else:  # Compare multiple files
        if 'multi_texts' not in st.session_state or not st.session_state.get('multi_texts'):
            st.error("No files loaded.")
            st.stop()

        texts = st.session_state['multi_texts']
        filenames = st.session_state['multi_filenames']

        with st.spinner("Computing similarities..."):
            sim_data = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    lex = lexical_similarity(texts[i], texts[j])
                    sem_raw = semantic_similarity(texts[i], texts[j])
                    sem_clamped = max(0.0, sem_raw)   # Fix for negative values in scatter size
                    sim_data.append({
                        'File 1': filenames[i],
                        'File 2': filenames[j],
                        'Lexical Sim (%)': round(lex * 100, 1),
                        'Semantic Sim (%)': round(sem_raw * 100, 1),
                        'Plot Size': round(sem_clamped * 100, 1)
                    })

            df = pd.DataFrame(sim_data)
            df = df.sort_values('Semantic Sim (%)', ascending=False)

            st.subheader("Pairwise File Similarities")
            st.dataframe(df[['File 1', 'File 2', 'Lexical Sim (%)', 'Semantic Sim (%)']], 
                         use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(df, x='File 1', y='Semantic Sim (%)', color='File 2',
                                 title='Semantic Similarity Bar', hover_data=['File 2'])
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_scatter = px.scatter(df, x='File 1', y='File 2',
                                         size='Plot Size',
                                         color='Semantic Sim (%)',
                                         hover_name='File 2',
                                         hover_data=['Lexical Sim (%)', 'Semantic Sim (%)'],
                                         title='Semantic Similarity Scatter')
                st.plotly_chart(fig_scatter, use_container_width=True)

st.info("""
TrueSource uses **semantic embeddings** (all-MiniLM-L6-v2) → detects paraphrases much better than classic methods.  
Google search is powered by Serper API.
""")