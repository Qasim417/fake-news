import json
import os
import re
from io import BytesIO

import PyPDF2
import streamlit as st
from groq import Groq
from tavily import TavilyClient

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
# SECRETS / API KEYS
# =========================
def get_secret(name: str) -> str:
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, "")

GROQ_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

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

def safe_parse_json(text: str):
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    candidate = candidate.strip().strip("```").replace("json", "", 1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def groq_json(prompt: str, system_msg: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    data = safe_parse_json(raw)
    return data, raw

def extract_claims(article_text: str):
    prompt = f"""
Extract up to 3 key factual claims from the news text below.
For each claim, create a short web search query.

Return VALID JSON only in this format:
{{
  "claims": [
    {{"claim": "short claim", "query": "search query"}},
    {{"claim": "short claim", "query": "search query"}}
  ]
}}

Rules:
- Keep claims short.
- Search query should be concise and factual.
- If text is weak or short, still return the best possible claims.
- Return JSON only.

News text:
{article_text[:7000]}
""".strip()

    data, raw = groq_json(
        prompt,
        "You are a JSON-only assistant that extracts factual claims.",
    )

    if not data or "claims" not in data:
        fallback = article_text[:250].strip()
        return [{"claim": fallback, "query": fallback}], raw

    claims = []
    for item in data.get("claims", [])[:3]:
        claim = str(item.get("claim", "")).strip()
        query = str(item.get("query", claim)).strip()
        if claim:
            claims.append({"claim": claim, "query": query[:300] or claim[:300]})

    if not claims:
        fallback = article_text[:250].strip()
        claims = [{"claim": fallback, "query": fallback}]

    return claims, raw

def search_web_for_claim(query: str):
    try:
        res = tavily_client.search(query=query)
        return res if isinstance(res, dict) else {}
    except Exception as e:
        return {"error": str(e), "results": []}

def gather_evidence(claims):
    evidence = []
    source_links = []

    for c in claims:
        query = c["query"]
        search_res = search_web_for_claim(query)
        results = search_res.get("results", []) or []

        items = []
        for r in results[:3]:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "") or r.get("snippet", "")
            if url:
                source_links.append(url)
            items.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet[:300],
                }
            )

        evidence.append(
            {
                "claim": c["claim"],
                "query": query,
                "answer": search_res.get("answer", ""),
                "items": items,
            }
        )

    unique_links = list(dict.fromkeys([u for u in source_links if u]))
    return evidence, unique_links

def final_assessment(article_text: str, source_url: str, claims, evidence):
    evidence_block = []
    for idx, block in enumerate(evidence, start=1):
        evidence_block.append(f"Claim {idx}: {block['claim']}")
        evidence_block.append(f"Search query: {block['query']}")
        if block.get("answer"):
            evidence_block.append(f"Search answer: {block['answer']}")
        for src in block.get("items", []):
            evidence_block.append(
                f"- Source: {src['title']} | {src['url']} | Snippet: {src['snippet']}"
            )
        evidence_block.append("")

    prompt = f"""
You are a careful fact-checking assistant.

Decide whether the news is:
- Real
- Fake
- Unverified

Use ONLY the article text and the search evidence below.

Return VALID JSON only with:
{{
  "verdict": "Real/Fake/Unverified",
  "confidence": 0-100,
  "explanation": "simple English explanation",
  "suspicious_claims": ["..."],
  "source_notes": ["..."],
  "final_note": "short closing note"
}}

Rules:
- If evidence supports the main claim, mark Real.
- If evidence contradicts the main claim, mark Fake.
- If evidence is weak, mixed, or missing, mark Unverified.
- Mention source URLs in source_notes.
- Keep explanation short but useful.
- Do not invent sources.

Article text:
{article_text[:9000]}

Optional user source URL:
{source_url if source_url else "None"}

Claims:
{json.dumps(claims, ensure_ascii=False)}

Search evidence:
{chr(10).join(evidence_block)}
""".strip()

    data, raw = groq_json(
        prompt,
        "You are a JSON-only fact-checking assistant.",
    )

    if not data:
        data = {
            "verdict": "Unverified",
            "confidence": 40,
            "explanation": raw or "No structured response returned.",
            "suspicious_claims": [],
            "source_notes": [],
            "final_note": "The model did not return valid JSON.",
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

    return data, raw

def verdict_badge(verdict: str):
    v = (verdict or "").lower()
    if v == "real":
        return '<span class="badge badge-real">REAL</span>'
    if v == "fake":
        return '<span class="badge badge-fake">FAKE</span>'
    return '<span class="badge badge-unknown">UNVERIFIED</span>'

def make_report(data: dict, source_url: str, evidence, input_excerpt: str):
    lines = []
    lines.append("# AI Fake News Detector Report")
    lines.append(f"- Verdict: {data.get('verdict', 'Unverified')}")
    lines.append(f"- Confidence: {data.get('confidence', 50)}%")
    lines.append("")
    lines.append("## Explanation")
    lines.append(str(data.get("explanation", "")))
    lines.append("")
    lines.append("## Suspicious Claims")
    suspicious = data.get("suspicious_claims", []) or []
    if suspicious:
        for item in suspicious:
            lines.append(f"- {item}")
    else:
        lines.append("- None detected")
    lines.append("")
    lines.append("## Source Notes")
    source_notes = data.get("source_notes", []) or []
    if source_notes:
        for item in source_notes:
            lines.append(f"- {item}")
    else:
        lines.append("- No source notes returned")
    lines.append("")
    lines.append("## Optional User Source URL")
    lines.append(source_url if source_url else "Not provided")
    lines.append("")
    lines.append("## Web Evidence")
    for idx, block in enumerate(evidence, start=1):
        lines.append(f"### Claim {idx}")
        lines.append(f"- Claim: {block['claim']}")
        lines.append(f"- Query: {block['query']}")
        if block.get("answer"):
            lines.append(f"- Search answer: {block['answer']}")
        for src in block.get("items", []):
            lines.append(f"- Source: {src['title']} | {src['url']}")
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
        "Article URL (optional)",
        placeholder="Paste the news link here",
        help="Optional: if you have the article link, add it here.",
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
        <p>Paste news text or upload a PDF/TXT file. The app checks claims with web search and then uses Groq to decide Real, Fake, or Unverified.</p>
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
    st.write("• Web sources from Tavily")
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
        with st.spinner("Checking claims with Tavily and Groq..."):
            claims, claims_raw = extract_claims(combined_text[:max_chars])
            evidence, unique_sources = gather_evidence(claims)
            data, raw = final_assessment(
                article_text=combined_text[:max_chars],
                source_url=source_url.strip(),
                claims=claims,
                evidence=evidence,
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
        top3.metric("Sources found", len(unique_sources))

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
            st.info("No source notes were returned.")

        st.markdown("### Web Sources")
        if unique_sources:
            for link in unique_sources:
                st.markdown(f"- {link}")
        else:
            st.warning("No web sources were found for the claims.")

        with st.expander("Show claim-by-claim evidence"):
            for idx, block in enumerate(evidence, start=1):
                st.markdown(f"**Claim {idx}:** {block['claim']}")
                st.markdown(f"**Query:** {block['query']}")
                if block.get("answer"):
                    st.markdown(f"**Search answer:** {block['answer']}")
                if block.get("items"):
                    for src in block["items"]:
                        st.markdown(
                            f"- **{src['title']}**  \n  {src['url']}  \n  {src['snippet']}"
                        )
                st.divider()

        st.markdown("### Final Note")
        st.write(data.get("final_note", ""))

        report = make_report(
            data=data,
            source_url=source_url.strip(),
            evidence=evidence,
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
