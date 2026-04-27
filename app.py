import json
import os
import re
from io import BytesIO

import PyPDF2
import streamlit as st
from groq import Groq

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# LIGHT UI STYLING
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #111827;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #eef4ff);
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        color: #111827;
    }

    .hero p {
        margin: 8px 0 0 0;
        color: #6b7280;
        font-size: 1rem;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(17, 24, 39, 0.06);
        margin-bottom: 16px;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }

    .badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        margin-right: 8px;
    }

    .badge-real {
        background: #dcfce7;
        color: #166534;
    }

    .badge-fake {
        background: #fee2e2;
        color: #991b1b;
    }

    .badge-unknown {
        background: #fef3c7;
        color: #92400e;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    .source-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SECRETS / API KEY
# =========================
def get_api_key() -> str:
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY", "")


API_KEY = get_api_key()

if not API_KEY:
    st.error(
        "GROQ_API_KEY missing. Add it in `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
    )
    st.stop()

client = Groq(api_key=API_KEY)

# Updated Groq production model
MODEL_NAME = "llama-3.3-70b-versatile"

# =========================
# HELPERS
# =========================
URL_REGEX = r"https?://[^\s\)\]\}>\"']+"

def extract_urls(text: str):
    return list(dict.fromkeys(re.findall(URL_REGEX, text or "")))

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    raw = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(raw))
        pages = []
        for page in pdf_reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    if file_name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore").strip()

    return ""

def build_prompt(news_text: str, source_url: str, detected_urls: list[str]) -> str:
    return f"""
You are a careful fake-news analysis assistant.

Task:
- Classify the input as exactly one of: Real, Fake, Unverified
- Give a confidence score from 0 to 100
- Explain in simple English
- Mention suspicious claims if any
- Mention source notes based only on provided input
- If there is not enough evidence, say Unverified
- Do NOT pretend you checked the live web unless a real source URL is provided in the input
- Keep the response concise and useful

Return VALID JSON only with these keys:
- verdict (string)
- confidence (integer)
- explanation (string)
- suspicious_claims (array of strings)
- source_notes (array of strings)
- final_note (string)

Input text:
{news_text[:12000]}

User-provided source URL:
{source_url if source_url else "None"}

URLs detected inside the provided content:
{detected_urls if detected_urls else "None"}
""".strip()

def safe_parse_json(text: str):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    candidate = candidate.strip().strip("```").replace("json", "", 1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def analyze_news(news_text: str, source_url: str):
    news_text = clean_text(news_text)
    detected_urls = extract_urls(news_text)

    prompt = build_prompt(news_text, source_url, detected_urls)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise fact-checking assistant that responds in valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        data = safe_parse_json(raw)

        if not data:
            data = {
                "verdict": "Unverified",
                "confidence": 50,
                "explanation": raw,
                "suspicious_claims": [],
                "source_notes": [],
                "final_note": "The model returned non-JSON output, so the app showed the raw answer.",
            }

        data.setdefault("verdict", "Unverified")
        data.setdefault("confidence", 50)
        data.setdefault("explanation", "")
        data.setdefault("suspicious_claims", [])
        data.setdefault("source_notes", [])
        data.setdefault("final_note", "")

        try:
            data["confidence"] = max(0, min(100, int(data["confidence"])))
        except Exception:
            data["confidence"] = 50

        return data, detected_urls, raw

    except Exception as e:
        return {
            "verdict": "Unverified",
            "confidence": 0,
            "explanation": f"Groq error: {e}",
            "suspicious_claims": [],
            "source_notes": [],
            "final_note": "The request could not be completed.",
        }, detected_urls, ""

def verdict_badge(verdict: str):
    v = (verdict or "").lower()
    if v == "real":
        return '<span class="badge badge-real">REAL</span>'
    if v == "fake":
        return '<span class="badge badge-fake">FAKE</span>'
    return '<span class="badge badge-unknown">UNVERIFIED</span>'

def make_report(data: dict, source_url: str, detected_urls: list[str], input_excerpt: str):
    suspicious = data.get("suspicious_claims", []) or []
    sources = data.get("source_notes", []) or []

    lines = []
    lines.append("# AI Fake News Detector Report")
    lines.append(f"- Verdict: {data.get('verdict', 'Unverified')}")
    lines.append(f"- Confidence: {data.get('confidence', 50)}%")
    lines.append("")
    lines.append("## Explanation")
    lines.append(str(data.get("explanation", "")))
    lines.append("")
    lines.append("## Suspicious Claims")
    if suspicious:
        for item in suspicious:
            lines.append(f"- {item}")
    else:
        lines.append("- None detected")
    lines.append("")
    lines.append("## Source Notes")
    if sources:
        for item in sources:
            lines.append(f"- {item}")
    else:
        lines.append("- No extra source notes returned")
    lines.append("")
    lines.append("## Source URL")
    lines.append(source_url if source_url else "Not provided")
    lines.append("")
    lines.append("## URLs Detected in Input")
    if detected_urls:
        for u in detected_urls:
            lines.append(f"- {u}")
    else:
        lines.append("- None detected")
    lines.append("")
    lines.append("## Input Excerpt")
    lines.append(input_excerpt[:2000])
    lines.append("")
    lines.append("## Final Note")
    lines.append(str(data.get("final_note", "")))

    return "\n".join(lines)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Settings")
    source_url = st.text_input(
        "Source URL (optional)",
        placeholder="Paste the article link here",
        help="This will be shown in the report and used as a source hint.",
    )
    show_raw = st.checkbox("Show raw AI output", value=False)
    max_chars = st.slider("Max characters sent to AI", 1000, 12000, 6000, 500)

# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>📰 AI Fake News Detector</h1>
        <p>Paste news text or upload a PDF/TXT file, then get a verdict, confidence score, explanation, suspicious claims, and source notes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =========================
# MAIN UI
# =========================
col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1) Input News</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Paste news text here",
        height=260,
        placeholder="Paste the news article, headline, or content here...",
    )

    uploaded_file = st.file_uploader(
        "Or upload a PDF/TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2) What this app gives</div>', unsafe_allow_html=True)
    st.write("• Real / Fake / Unverified verdict")
    st.write("• Confidence score")
    st.write("• Simple explanation")
    st.write("• Suspicious claims list")
    st.write("• Source notes and detected URLs")
    st.write("• Downloadable report")
    st.markdown("</div>", unsafe_allow_html=True)

input_text = text_input.strip()
file_text = read_uploaded_file(uploaded_file)

if input_text and file_text:
    combined_text = input_text + "\n\n" + file_text
elif file_text:
    combined_text = file_text
else:
    combined_text = input_text

analyze_btn = st.button("🔍 Analyze News", use_container_width=True)

if analyze_btn:
    if not combined_text.strip():
        st.warning("Please paste some news text or upload a PDF/TXT file.")
    else:
        with st.spinner("Analyzing with Groq..."):
            data, detected_urls, raw = analyze_news(
                combined_text[:max_chars],
                source_url.strip(),
            )

        verdict = data.get("verdict", "Unverified")
        confidence = data.get("confidence", 50)

        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">Result</div>
                {verdict_badge(verdict)}
                <div style="height:12px;"></div>
            """,
            unsafe_allow_html=True,
        )

        top1, top2, top3 = st.columns(3)
        top1.metric("Verdict", verdict)
        top2.metric("Confidence", f"{confidence}%")
        top3.metric("Detected URLs", len(detected_urls))

        st.progress(confidence / 100.0)

        st.markdown("### Explanation")
        st.write(data.get("explanation", ""))

        st.markdown("### Suspicious Claims")
        suspicious_claims = data.get("suspicious_claims", []) or []
        if suspicious_claims:
            for item in suspicious_claims:
                st.write(f"• {item}")
        else:
            st.info("No suspicious claim was highlighted by the model.")

        st.markdown("### Source Notes")
        source_notes = data.get("source_notes", []) or []
        if source_notes:
            for item in source_notes:
                st.write(f"• {item}")
        else:
            st.info("No additional source notes were returned.")

        st.markdown("### Source Links")
        source_link_items = []
        if source_url.strip():
            source_link_items.append(source_url.strip())
        source_link_items.extend(detected_urls)

        if source_link_items:
            unique_links = list(dict.fromkeys(source_link_items))
            for link in unique_links:
                st.markdown(f"- {link}")
        else:
            st.warning("No source link found in the input. Add a source URL for a stronger report.")

        st.markdown("### Final Note")
        st.write(data.get("final_note", ""))

        report = make_report(
            data=data,
            source_url=source_url.strip(),
            detected_urls=detected_urls,
            input_excerpt=combined_text[:max_chars],
        )

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="fake_news_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if show_raw and raw:
            st.markdown("### Raw AI Output")
            st.code(raw, language="json")

        st.markdown("</div>", unsafe_allow_html=True)
