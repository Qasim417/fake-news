import json
import os
import re
from io import BytesIO

import PyPDF2
import streamlit as st
from groq import Groq

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ==============================
# LIGHT THEME STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #1f2937;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

/* Header */
.title {
    font-size: 32px;
    font-weight: bold;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    font-size: 15px;
}

/* Badges */
.badge {
    padding: 8px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 14px;
}

.real {background:#dcfce7; color:#166534;}
.fake {background:#fee2e2; color:#991b1b;}
.unknown {background:#fef3c7; color:#92400e;}

</style>
""", unsafe_allow_html=True)

# ==============================
# API KEY
# ==============================
def get_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY")

client = Groq(api_key=get_api_key())

# ==============================
# HELPERS
# ==============================
def extract_text(file):
    reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_news(text):
    prompt = f"""
    Detect fake news.

    Give:
    - Verdict (Real/Fake/Unverified)
    - Confidence (0-100)
    - Explanation
    - Suspicious points

    Text:
    {text[:4000]}
    """

    res = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content

def get_badge(result):
    result = result.lower()
    if "real" in result:
        return '<span class="badge real">REAL</span>'
    elif "fake" in result:
        return '<span class="badge fake">FAKE</span>'
    else:
        return '<span class="badge unknown">UNVERIFIED</span>'

# ==============================
# UI
# ==============================
st.markdown('<div class="title">📰 AI Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check if news is real or fake using AI</div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter News")

    text_input = st.text_area("Paste news text")

    file = st.file_uploader("Upload PDF", type=["pdf"])

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Features")
    st.write("✔ Detect Real / Fake news")
    st.write("✔ Confidence estimation")
    st.write("✔ Explanation")
    st.write("✔ File support (PDF)")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PROCESS
# ==============================
if st.button("Analyze News"):

    if file:
        text = extract_text(file)
    else:
        text = text_input

    if not text:
        st.warning("Enter text or upload file")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_news(text)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(get_badge(result), unsafe_allow_html=True)

        st.subheader("Result")
        st.write(result)

        st.markdown('</div>', unsafe_allow_html=True)
