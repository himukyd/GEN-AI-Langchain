import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv
import re
import time

load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Brain",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark cinematic background ── */
.stApp {
    background: linear-gradient(180deg, #f7f8fc 0%, #eef2f9 100%);
    color: #1f2430;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d9dfeb;
}
[data-testid="stSidebar"] * {
    color: #2a3242 !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── Main title ── */
.yt-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff4d6d 0%, #ff9a3c 50%, #ffcf57 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.yt-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #5b6476;
    font-weight: 400;
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
}

/* ── URL input ── */
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #cfd7e6 !important;
    border-radius: 12px !important;
    color: #1f2430 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput input:focus {
    border-color: #ff4d6d !important;
    box-shadow: 0 0 0 3px rgba(255,77,109,0.10) !important;
}
.stTextInput label {
    color: #5b6476 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #ff4d6d, #ff6b35) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(255,77,109,0.35) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(185, 196, 214, 0.55) !important;
    border-radius: 18px !important;
    padding: 0.2rem 0.35rem !important;
    margin-bottom: 0.55rem !important;
    backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"] * {
    color: #1f2430 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li {
    color: #1f2430 !important;
    font-size: 1.02rem !important;
    line-height: 1.65 !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #f7f3ee !important;
    width: 100% !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    transform: scale(1.02);
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(255, 77, 109, 0.12), rgba(255, 107, 53, 0.06)) !important;
    border: 1px solid rgba(255, 77, 109, 0.22) !important;
    box-shadow: 0 10px 22px rgba(255, 77, 109, 0.05) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.85rem 1.2rem !important;
    max-width: 80% !important;
    margin-left: auto !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255, 255, 255, 0.94) !important;
    border: 1px solid rgba(185, 196, 214, 0.60) !important;
    box-shadow: 0 10px 22px rgba(31, 36, 48, 0.06) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.85rem 1.2rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li {
    color: #1f2430 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li {
    color: #1f2430 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1px solid #cfd7e6 !important;
    border-radius: 16px !important;
    color: #1f2430 !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: inset 0 1px 0 rgba(31,36,48,0.03) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #7a8496 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #ff4d6d !important;
    box-shadow: 0 0 0 3px rgba(255,77,109,0.10) !important;
}
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.92) !important;
    border-top: 1px solid #d9dfeb !important;
    padding-top: 0.4rem !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #ff4d6d, #ff6b35) !important;
    color: white !important;
    border-radius: 14px !important;
}

.thinking-bubble {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-weight: 600;
    color: #5b6476;
}
.thinking-dots {
    display: inline-flex;
    gap: 0.15rem;
}
.thinking-dots span {
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 50%;
    background: #ff4d6d;
    animation: thinkingPulse 1.1s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) {
    animation-delay: 0.15s;
}
.thinking-dots span:nth-child(3) {
    animation-delay: 0.3s;
}
@keyframes thinkingPulse {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
    40% { transform: translateY(-3px); opacity: 1; }
}

/* ── Info / success / error boxes ── */
.stAlert {
    background: #ffffff !important;
    border-radius: 12px !important;
    border-left: 3px solid #ff4d6d !important;
    color: #2a3242 !important;
}

/* ── Divider ── */
hr {
    border-color: #d9dfeb !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #ff4d6d !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #eef2f9; }
::-webkit-scrollbar-thumb { background: #c4ccda; border-radius: 2px; }

/* ── Video info card ── */
.video-card {
    background: #ffffff;
    border: 1px solid #d9dfeb;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.video-card .vid-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #ff4d6d;
    font-weight: 600;
    margin-bottom: 0.3rem;
    font-family: 'Syne', sans-serif;
}
.video-card .vid-id {
    font-size: 0.9rem;
    color: #627085;
    font-family: 'DM Mono', monospace;
    word-break: break-all;
}

/* ── Chunk stats ── */
.stat-pill {
    display: inline-block;
    background: #f3f6fb;
    border: 1px solid #d9dfeb;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #5b6476;
    font-family: 'DM Sans', sans-serif;
    margin-right: 0.5rem;
}
.stat-pill span {
    color: #ff9a3c;
    font-weight: 600;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #627085;
}
.empty-state .big-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-state p {
    font-size: 0.95rem;
    font-family: 'DM Sans', sans-serif;
}

</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from any common URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    # If it already looks like a bare ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", url.strip()):
        return url.strip()
    return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def youtube_embed_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


@st.cache_resource(show_spinner=False)
def build_chain(video_id: str):
    """Ingest transcript and build RAG chain. Cached per video_id."""
    # Models
    llm      = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Transcript
    ytt_api       = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    transcript    = " ".join(chunk.text for chunk in transcript_list)

    # Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks   = splitter.create_documents([transcript])

    # Vector store (in-memory, fresh each session)
    vector_store = Chroma(
        collection_name=f"yt_{video_id}",
        embedding_function=embedder,
    )
    vector_store.reset_collection()
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = PromptTemplate(
        template="""You are a helpful YouTube transcript assistant.
    Answer ONLY from the provided transcript context below.
    Be concise, clear, and helpful.

    If the user input is vague, a greeting, or cannot be answered from the transcript, do not reply with only "I don't know".
    Instead, respond in this format:
    1. Ask the user to ask a specific question based on the provided transcript.
    2. Give one example of a better question they can ask.
    3. Add one short LLM response summarizing what the transcript section seems to be about.

    Keep the tone friendly and useful.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    chain_1 = RunnableParallel(
        context  = retriever | RunnableLambda(format_docs),
        question = RunnablePassthrough(),
    ) | prompt

    chain_2 = llm | StrOutputParser()

    return chain_1 | chain_2, len(chunks)


# ─── Session state ───────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "chain"      not in st.session_state: st.session_state.chain      = None
if "video_id"   not in st.session_state: st.session_state.video_id   = None
if "num_chunks" not in st.session_state: st.session_state.num_chunks = 0
if "loaded"     not in st.session_state: st.session_state.loaded     = False
if "sidebar_visible" not in st.session_state: st.session_state.sidebar_visible = True


def set_sidebar_visible(visible: bool) -> None:
    st.session_state.sidebar_visible = visible


if not st.session_state.sidebar_visible:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            border: none !important;
        }
        [data-testid="stSidebarContent"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.session_state.loaded and st.session_state.sidebar_visible:
        st.button("Hide menu", key="hide_menu_btn", on_click=set_sidebar_visible, args=(False,))

    st.markdown('<p class="yt-title">YT Brain</p>', unsafe_allow_html=True)
    st.markdown('<p class="yt-subtitle">Ask anything about any YouTube video</p>', unsafe_allow_html=True)

    st.markdown("---")

    url_input = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=...",
        key="url_input",
    )

    load_btn = st.button("⚡  Load Video", key="load_btn")

    if load_btn and url_input:
        vid_id = extract_video_id(url_input)
        if not vid_id:
            st.error("Couldn't find a valid video ID. Please check the URL.")
        else:
            with st.spinner("Fetching transcript and building index…"):
                try:
                    chain, n_chunks = build_chain(vid_id)
                    st.session_state.chain      = chain
                    st.session_state.video_id   = vid_id
                    st.session_state.num_chunks = n_chunks
                    st.session_state.messages   = []
                    st.session_state.loaded     = True
                    st.session_state.sidebar_visible = False
                    st.success("Video loaded! Start asking questions below.")
                except TranscriptsDisabled:
                    st.error("This video has no captions available.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Video info card
    if st.session_state.loaded:
        st.markdown("---")
        st.markdown(f"""
        <div class="video-card">
            <div class="vid-label">Active Video</div>
            <div class="vid-id">{st.session_state.video_id}</div>
        </div>
        <div>
            <span class="stat-pill">Chunks <span>{st.session_state.num_chunks}</span></span>
            <span class="stat-pill">k = <span>4</span></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑  Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # How it works
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#3a3848; line-height:1.8;">
    <strong style="color:#2a3242; font-family:'Syne',sans-serif;">How it works</strong><br>
    1. Paste any YouTube URL<br>
    2. Transcript is chunked + embedded<br>
    3. Your question retrieves top-4 chunks<br>
    4. Gemini answers from context only
    </div>
    """, unsafe_allow_html=True)


# ─── Main area ───────────────────────────────────────────────────────────────
if not st.session_state.loaded:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">🎬</div>
        <p>Paste a YouTube URL in the sidebar<br>and hit <strong>Load Video</strong> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if not st.session_state.sidebar_visible:
        if st.button("☰ Show menu", key="show_menu_btn"):
            st.session_state.sidebar_visible = True
            st.rerun()

    top_left, top_right = st.columns([1.6, 1], gap="large")

    with top_left:
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <div style="font-family:'Syne',sans-serif;font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;color:#d9485f;margin-bottom:0.35rem;">
                    Now playing
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:#1f2430;line-height:1.15;">
                    Watch the video, then ask questions below
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.video(youtube_embed_url(st.session_state.video_id))

    with top_right:
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.96);border:1px solid #d9dfeb;border-radius:18px;padding:1rem 1.1rem;margin-bottom:0.9rem;box-shadow:0 8px 24px rgba(31,36,48,0.05);">
                <div style="font-family:'Syne',sans-serif;font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;color:#627085;margin-bottom:0.35rem;">
                    Chat bot
                </div>
                <div style="font-size:1rem;font-weight:700;color:#1f2430;margin-bottom:0.35rem;word-break:break-word;">
                    Ask questions about this video
                </div>
                <div style="font-size:0.88rem;line-height:1.6;color:#5b6476;">
                    <span style="color:#d9485f;font-weight:700;">{st.session_state.num_chunks} chunks</span> indexed from <span style="color:#1f2430;">{st.session_state.video_id}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chat_history = st.container(height=560, border=False)

        with chat_history:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if question := st.chat_input("Ask anything about this video…"):
            st.session_state.messages.append({"role": "user", "content": question})

            with chat_history:
                with st.chat_message("assistant"):
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown(
                        '<div class="thinking-bubble">Thinking <span class="thinking-dots"><span></span><span></span><span></span></span></div>',
                        unsafe_allow_html=True,
                    )

                    full_response = ""
                    try:
                        # Stream token by token
                        for chunk in st.session_state.chain.stream(question):
                            full_response += chunk
                            thinking_placeholder.markdown(full_response + "▌")
                        thinking_placeholder.markdown(full_response)
                    except Exception as e:
                        full_response = f"Something went wrong: {e}"
                        thinking_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()