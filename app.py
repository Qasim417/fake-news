import json
import os
import re
from io import BytesIO

import PyPDF2
import streamlit as st
from groq import Groq

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme styles
# -----------------------------
THEMES = {
    "Dark Blue": {
        "bg": "#0e1117",
        "card": "#161b22",
        "card2": "#1f2937",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "accent": "#60a5fa",
        "accent2": "#22c55e",
        "border": "rgba(255,255,255,0.10)",
    },
    "Purple Neon": {
        "bg": "#0b1020",
        "card": "#121932",
        "card2": "#1a2244",
        "text": "#f3f4f6",
        "muted": "#b7bfdc",
        "accent": "#a78bfa",
        "accent2": "#34d399",
        "border": "rgba(167,139,250,0.20)",
    },
    "Ocean": {
        "bg": "#071b2e",
        "card": "#0b243d",
        "card2": "#12324f",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "accent": "#38bdf8",
        "accent2": "#2dd4bf",
        "border": "rgba(56,189,248,0.18)",
    },
}

def apply_theme(theme_name: str):
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top, {t["card2"]} 0%, {t["bg"]} 55%);
                color: {t["text"]};
            }}
            .main-container {{
                padding-top: 0.5rem;
            }}
            .hero {{
                background: linear-gradient(135deg, {t["card"]}, {t["card2"]});
                border: 1px solid {t["border"]};
                border-radius: 22px;
                padding: 24px 24px 18px 24px;
                box-shadow: 0 12px 35px rgba(0,0,0,0.25);
            }}
            .hero h1 {{
                margin: 0;
                font-size: 2rem;
                color: {t["text"]};
            }}
            .hero p {{
                margin: 8px 0 0 0;
                color: {t["muted"]};
                font-size: 0.98rem;
            }}
            .card {{
                background: linear-gradient(180deg, {t["card"]}, {t["card2"]});
                border: 1px solid {t["border"]};
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.18);
            }}
            .small-label {{
                color: {t["muted"]};
                font-size: 0.88rem;
                margin-bottom: 0.25rem;
            }}
            .badge {{
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.92rem;
                margin-right: 8px;
                border: 1px solid {t["border"]};
            }}
            .badge-real {{
                background: rgba(34,197,94,0.12);
                color: #86efac;
            }}
            .badge-fake {{
                background: rgba(239,68,68,0.12);
                color: #fca5a5;
            }}
            .badge-unknown {{
                background: rgba(245,158,11,0.12);
                color: #fcd34d;
            }}
            .source-box {{
                background: rgba(255,255,255,0.04);
                border: 1px solid {t["border"]};
                border-radius: 14px;
                padding: 12px 14px;
                margin-top: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Secrets / API key
# -----------------------------
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

# Groq production model from the docs
MODEL_NAME = "llama-3.3-70b-versatile"

# -----------------------------
# Helpers
# -----------------------------
URL_REGEX = r"https?://[^\s\)\]\}>\"']+"

def extract_urls(text: str):
    return list(dict.fromkeys(re.findall(URL_REGEX, text or "")))

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(raw))
        pages_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        return "\n".join(pages_text).strip()

    if name.endswith(".txt"):
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

    # keep confidence safe
    try:
        data["confidence"] = max(0, min(100, int(data["confidence"])))
    except Exception:
        data["confidence"] = 50

    return data, detected_urls, raw

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

    report = []
    report.append("# AI Fake News Detector Report")
    report.append(f"- Verdict: {data.get('verdict', 'Unverified')}")
    report.append(f"- Confidence: {data.get('confidence', 50)}%")
    report.append("")
    report.append("## Explanation")
    report.append(str(data.get("explanation", "")))
    report.append("")
    report.append("## Suspicious Claims")
    if suspicious:
        for item in suspicious:
            report.append(f"- {item}")
    else:
        report.append("- None detected")
    report.append("")
    report.append("## Source Notes")
    if sources:
        for item in sources:
            report.append(f"- {item}")
    else:
        report.append("- No extra source notes returned")
    report.append("")
    report.append("## Source URL")
    report.append(source_url if source_url else "Not provided")
    report.append("")
    report.append("## URLs Detected in Input")
    if detected_urls:
        for u in detected_urls:
            report.append(f"- {u}")
    else:
        report.append("- None detected")
    report.append("")
    report.append("## Input Excerpt")
    report.append(input_excerpt[:2000])
    report.append("")
    report.append("## Final Note")
    report.append(str(data.get("final_note", "")))
    return "\n".join(report)

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)
    st.caption("Groq key is loaded from secrets.")
    source_url = st.text_input(
        "Source URL (optional)",
        placeholder="Paste the article link here",
        help="This helps the model treat the article as a source and list it in the report.",
    )
    show_raw = st.checkbox("Show raw AI output", value=False)
    max_chars = st.slider("Max input characters sent to AI", 1000, 12000, 6000, 500)

apply_theme(theme_name)

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

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Input News")
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
    st.subheader("2) What this app gives")
    st.write("• Real / Fake / Unverified verdict")
    st.write("• Confidence score")
    st.write("• Simple explanation")
    st.write("• Suspicious claims list")
    st.write("• Source notes and detected URLs")
    st.write("• Downloadable report")
    st.markdown("</div>", unsafe_allow_html=True)

input_text = text_input.strip()
file_text = read_uploaded_file(uploaded_file)

combined_text = ""
if input_text and file_text:
    combined_text = input_text + "\n\n" + file_text
elif file_text:
    combined_text = file_text
else:
    combined_text = input_text

st.write("")

analyze_btn = st.button("🔍 Analyze News", use_container_width=True)

if analyze_btn:
    if not combined_text.strip():
        st.warning("Please paste some news text or upload a PDF/TXT file.")
    else:
        with st.spinner("Analyzing with Groq..."):
            try:
                data, detected_urls, raw = analyze_news(
                    combined_text[:max_chars],
                    source_url.strip(),
                )

                verdict = data.get("verdict", "Unverified")
                confidence = data.get("confidence", 50)

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="small-label">Result</div>
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
                    st.warning("No source link found in the input. Add a source URL for stronger checking.")

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

                if show_raw:
                    st.markdown("### Raw AI Output")
                    st.code(raw, language="json")

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
