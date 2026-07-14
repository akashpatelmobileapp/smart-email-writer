from dotenv import load_dotenv
load_dotenv()

import os
import re
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from tavily import TavilyClient

st.set_page_config(page_title="MailCraft AI", page_icon="✉️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #09090f !important;
    color: #ffffff;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
/* Hide sidebar collapse/hide button */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"][aria-label*="sidebar"] { display: none !important; }
/* Force sidebar always visible, even if collapsed */
[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    min-width: 21rem !important;
    width: 21rem !important;
    margin-left: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
    transform: none !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1a1a2e !important;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem !important; }

.stTextInput label, .stSelectbox label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #aaaacc !important;
}
.stTextInput input {
    background: #12121e !important;
    border: 1px solid #1e1e32 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stTextInput input::placeholder { color: #252535 !important; }
.stSelectbox > div > div {
    background: #12121e !important;
    border: 1px solid #1e1e32 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

div[data-testid="stButton"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    font-size: 0.86rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    padding: 0.72rem 1.5rem !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    opacity: 0.85 !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
    background: #12121e !important;
    color: #9090b8 !important;
    border: 1px solid #1e1e32 !important;
    padding: 0.62rem 1rem !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #6366f1 !important;
    color: #ffffff !important;
}
div[data-testid="stDownloadButton"] button {
    background: #12121e !important;
    color: #9090b8 !important;
    border: 1px solid #1e1e32 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    width: auto !important;
}

/* Hide default spinner text */
.stSpinner p { display: none !important; }
/* Hide "Press Enter to apply" hint */
.stTextInput small { display: none !important; }
[data-testid="InputInstructions"] { display: none !important; }
.stTextInput > div > div > div > small { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────
def linkify(text: str) -> str:
    """Convert URLs in text to clickable HTML links."""
    url_pattern = re.compile(r'(https?://[^\s\)\]\>]+)')
    return url_pattern.sub(
        r'<a href="\1" target="_blank" style="color:#ffffff;text-decoration:underline;text-underline-offset:2px">\1</a>',
        text
    )

def format_attachments(raw: str) -> str:
    """Parse attachment text into clean uniform HTML list items."""
    lines = raw.strip().split("\n")
    items = []
    current_title = ""
    current_body  = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # numbered item like "1. Something"
        if re.match(r'^\d+\.', line):
            if current_title:
                items.append((current_title, current_body.strip()))
            current_title = re.sub(r'^\d+\.\s*', '', line)
            current_body  = ""
        elif line.startswith("📎"):
            continue  # skip header line
        else:
            current_body += " " + line

    if current_title:
        items.append((current_title, current_body.strip()))

    if not items:
        # fallback: just render linkified text
        return f'<p style="font-size:0.88rem;color:#ffffff;line-height:1.7">{linkify(raw)}</p>'

    html = ""
    for i, (title, body) in enumerate(items):
        body_html = linkify(body)
        html += f"""
        <div style="display:flex;gap:14px;padding:1rem 0;
                    {'border-top:1px solid #1a1a2e' if i>0 else ''}">
            <div style="width:28px;height:28px;background:rgba(99,102,241,0.12);
                        border:1px solid rgba(99,102,241,0.25);border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.75rem;font-weight:700;color:#ffffff;flex-shrink:0;
                        margin-top:2px">{i+1}</div>
            <div style="flex:1">
                <div style="font-size:0.88rem;font-weight:600;color:#ffffff;margin-bottom:4px">{title}</div>
                <div style="font-size:0.82rem;color:#ffffff;line-height:1.65">{body_html}</div>
            </div>
        </div>"""
    return html

# ── Session state ─────────────────────────────────────────
for k, v in [("final_email", None), ("topic", ""), ("tone", "formal"), ("attachments", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── LangChain ─────────────────────────────────────────────
@st.cache_resource
def get_chain_and_agent():
    llm    = ChatMistralAI(model="mistral-small-2506")
    parser = StrOutputParser()
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    @tool
    def search_attachments(topic: str) -> str:
        """Search for relevant attachment suggestions for a given email topic."""
        response = tavily_client.search(
            query=f"relevant documents attachments for {topic} email",
            max_results=5,
        )
        results = response.get("results", [])
        if not results:
            return "No attachment suggestions found."
        out = []
        for i, r in enumerate(results):
            url     = r.get("url", "")
            snippet = r["content"][:130]
            out.append(f"{i+1}. {r['title']}\n   {snippet}...\n   {url}")
        return "\n\n".join(out)

    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional email writing assistant."),
        ("human",  "Write a {tone} email about: {topic}")
    ])
    grammar_prompt = ChatPromptTemplate.from_messages([
        ("system", "Fix grammar and improve tone. Return only the improved email."),
        ("human",  "{email}")
    ])
    shorten_prompt = ChatPromptTemplate.from_messages([
        ("system", "Shorten this email to under 150 words. Return only the shortened email."),
        ("human",  "{email}")
    ])
    chain = (
        draft_prompt | llm | parser
        | RunnableLambda(lambda x: {"email": x})
        | grammar_prompt | llm | parser
        | RunnableLambda(lambda x: {"email": x})
        | shorten_prompt | llm | parser
    )
    ag = create_agent(model=llm, tools=[search_attachments])
    return chain, ag

email_chain, agent = get_chain_and_agent()
has_email  = st.session_state.final_email  is not None
has_attach = st.session_state.attachments  is not None

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:2.5rem">
        <div style="width:40px;height:40px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px">✉️</div>
        <div>
            <div style="font-size:1.05rem;font-weight:700;color:#ffffff;letter-spacing:-0.02em">MailCraft AI</div>
            <div style="font-size:0.68rem;color:#aaaacc;margin-top:1px"></div>
        </div>
    </div>
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                color:#aaaacc;margin-bottom:0.9rem">Compose</div>
    """, unsafe_allow_html=True)

    topic = st.text_input("Topic", placeholder="e.g. project deadline extension",
                          value=st.session_state.topic, label_visibility="visible")
    tone  = st.selectbox("Tone", ["formal", "casual", "friendly"],
                         index=["formal","casual","friendly"].index(st.session_state.tone))
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    gen_btn = st.button("✦  Generate Email", type="primary", disabled=not topic.strip())

    # Pipeline
    st.markdown("""
    <div style="margin-top:2.5rem;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                text-transform:uppercase;color:#aaaacc;margin-bottom:1rem">Pipeline</div>
    """, unsafe_allow_html=True)

    def pip_step(num, label, desc, done):
        if done:
            dot = f'<div style="width:26px;height:26px;border-radius:50%;background:#6366f1;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:700;color:#fff;flex-shrink:0">✓</div>'
            tc, dc = "#ffffff", "#aaaacc"
        else:
            dot = f'<div style="width:26px;height:26px;border-radius:50%;border:2px solid #1e1e32;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;color:#aaaacc;flex-shrink:0">{num}</div>'
            tc, dc = "#aaaacc", "#666680"
        return f"""<div style="display:flex;align-items:flex-start;gap:12px;padding:8px 0">
            {dot}
            <div>
                <div style="font-size:0.83rem;font-weight:600;color:{tc}">{label}</div>
                <div style="font-size:0.72rem;color:{dc};margin-top:2px">{desc}</div>
            </div>
        </div>"""

    st.markdown(
        pip_step(1, "Draft + Grammar + Shorten", "3-step processing", has_email) +
        pip_step(2, "Attachment search", "Searching", has_attach),
        unsafe_allow_html=True
    )

    if has_email:
        st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
        if st.button("↩ Start over", type="secondary"):
            for k in ["final_email", "attachments"]:
                st.session_state[k] = None
            st.session_state.topic = ""
            st.session_state.tone  = "formal"
            st.rerun()

# ── Generate ──────────────────────────────────────────────
if gen_btn and topic.strip():
    with st.spinner(""):
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:1rem 1.4rem;
                    background:#12121e;border:1px solid #1e1e32;border-radius:12px;margin-bottom:1rem">
            <div style="width:8px;height:8px;border-radius:50%;background:#6366f1;
                        animation:pulse 1.2s infinite"></div>
            <span style="font-size:0.85rem;color:#ffffff">Generating your email...</span>
        </div>
        <style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}</style>
        """, unsafe_allow_html=True)
        st.session_state.final_email = email_chain.invoke({"topic": topic.strip(), "tone": tone})
        st.session_state.topic       = topic.strip()
        st.session_state.tone        = tone
        st.session_state.attachments = None
    st.rerun()

# ── MAIN CONTENT ──────────────────────────────────────────
_, col, _ = st.columns([1, 7, 1])

with col:
    st.markdown("<div style='padding-top:2.5rem'></div>", unsafe_allow_html=True)

    if not has_email:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    min-height:65vh;text-align:center">
            <div style="width:72px;height:72px;background:#0d0d1a;border:1px solid #1a1a2e;
                        border-radius:18px;display:flex;align-items:center;justify-content:center;
                        font-size:2rem;margin-bottom:1.2rem">✉️</div>
            <div style="font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:0.5rem">
                Your email will appear here
            </div>
            <div style="font-size:0.85rem;color:#aaaacc;max-width:280px;line-height:1.6">
                Fill in topic and tone in the sidebar,<br>then click
                <span style="color:#ffffff;font-weight:600">Generate Email</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        wc = len(st.session_state.final_email.split())

        # Header
        st.markdown(f"""
        <div style="margin-bottom:1.6rem">
            <div style="font-size:1.4rem;font-weight:700;color:#ffffff;letter-spacing:-0.02em;margin-bottom:0.4rem">
                Email Ready
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span style="font-size:0.78rem;color:#ffffff">Topic:</span>
                <span style="font-size:0.78rem;font-weight:600;color:#ffffff">{st.session_state.topic}</span>
                <span style="color:#1e1e32">·</span>
                <span style="font-size:0.78rem;color:#ffffff">Tone:</span>
                <span style="font-size:0.78rem;font-weight:600;color:#ffffff">{st.session_state.tone}</span>
                <span style="color:#1e1e32">·</span>
                <span style="font-size:0.72rem;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);
                              color:#ffffff;padding:2px 10px;border-radius:20px;font-weight:500">{wc} words</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Email card
        email_safe = (st.session_state.final_email
                      .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        st.markdown(f"""
        <div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:16px;
                    overflow:hidden;margin-bottom:1.2rem">
            <div style="display:flex;align-items:center;gap:8px;padding:0.85rem 1.4rem;
                        border-bottom:1px solid #1a1a2e;background:#10101e">
                <div style="width:8px;height:8px;border-radius:50%;background:#6366f1"></div>
                <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                             text-transform:uppercase;color:#aaaacc">Final Draft</span>
            </div>
            <div style="padding:1.8rem 2rem;font-size:0.92rem;line-height:1.85;
                        color:#ffffff;white-space:pre-wrap;font-family:'Inter',sans-serif">
{email_safe}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button("⬇  Download .txt", data=st.session_state.final_email,
                           file_name="email.txt", mime="text/plain")

        st.markdown("<div style='border-top:1px solid #1a1a2e;margin:2rem 0'></div>", unsafe_allow_html=True)

        # Attachments
        st.markdown("""
        <div style="margin-bottom:1.2rem">
            <div style="font-size:1rem;font-weight:700;color:#ffffff;margin-bottom:0.3rem">
                📎 Attachment Suggestions
            </div>
            <div style="font-size:0.82rem;color:#aaaacc;line-height:1.6">
                Let the agent find documents you might want to attach.
            </div>
        </div>
        """, unsafe_allow_html=True)

        attach_btn = st.button("🔍  Find relevant attachments", type="secondary")

        if attach_btn:
            with st.spinner(""):
                st.markdown("""
                <div style="display:flex;align-items:center;gap:10px;padding:1rem 1.4rem;
                            background:#12121e;border:1px solid #1e1e32;border-radius:12px;margin-bottom:1rem">
                    <div style="width:8px;height:8px;border-radius:50%;background:#6366f1;
                                animation:pulse 1.2s infinite"></div>
                    <span style="font-size:0.85rem;color:#ffffff">Searching for attachments...</span>
                </div>
                """, unsafe_allow_html=True)
                resp = agent.invoke({
                    "messages": [
                        SystemMessage(content="""You are an email assistant.
When user asks for attachment suggestions, call the search_attachments tool.
Only call the tool when explicitly asked."""),
                        HumanMessage(content=f"Email topic: {st.session_state.topic}\n\nUser request: add attachment suggestion")
                    ]
                })
                st.session_state.attachments = resp["messages"][-1].content
            st.rerun()

        if has_attach:
            formatted = format_attachments(st.session_state.attachments)
            st.markdown(f"""
            <div style="background:#0d0d1a;border:1px solid #1a1a2e;border-left:3px solid #6366f1;
                        border-radius:12px;overflow:hidden;margin-top:0.8rem">
                <div style="padding:0.8rem 1.4rem;border-bottom:1px solid #1a1a2e;background:#10101e;
                            font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#aaaacc">
                    📎 &nbsp;Suggested Attachments
                </div>
                <div style="padding:0.4rem 1.4rem 1rem">
                    {formatted}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)