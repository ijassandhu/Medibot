import streamlit as st
from memory_with_llm import get_answer
import base64

def load_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = load_image("images/logo.png")
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Medibot",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

html, body, [class*="css"]  {
    font-family: "Inter", sans-serif;
}

/* NAVBAR */
.navbar {
    width:100%;
    padding:14px 24px;
    background: #0f172a;
    border-bottom:1px solid #1e293b;
    display:flex;
    align-items:center;
    gap:10px;
    font-size:20px;
    font-weight:600;
    color:white;
}

/* CHAT CONTAINER */
.chat-container {
    max-width:900px;
    margin:auto;
}

/* USER MESSAGE */
.user-msg {
    background:#2563eb;
    color:white;
    padding:12px 16px;
    border-radius:18px;
    margin:10px 0;
    max-width:75%;
    margin-left:auto;
    box-shadow:0 2px 6px rgba(0,0,0,0.15);
}

/* BOT MESSAGE */
.bot-msg {
    background:#f1f5f9;
    color:#111827;
    padding:12px 16px;
    border-radius:18px;
    margin:10px 0;
    max-width:75%;
    margin-right:auto;
    border:1px solid #e2e8f0;
}

/* AVATAR ROW */
.msg-row {
    display:flex;
    align-items:flex-start;
    gap:10px;
}

.avatar {
    font-size:22px;
}

/* INPUT AREA */
.stChatInput {
    padding-bottom:20px;
}

/* SIDEBAR */
.sidebar-title {
    font-size:22px;
    font-weight:700;
}

.sidebar-desc {
    font-size:14px;
    color:#94a3b8;
}

.clear-btn {
    margin-top:20px;
}


</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVBAR
# -----------------------------

st.image("images/medibot_logo.png", width=200)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.image("images/name.png", width=150)
    st.caption("Medical assistant trained on the GALE Encyclopedia")

    st.markdown("---")

    # NEW CHAT
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💬 Chats")

    for title in st.session_state.conversations.keys():

        if st.button(title, key=title, use_container_width=True):

            st.session_state.current_chat = title
            st.session_state.messages = st.session_state.conversations[title]
            st.rerun()

    st.markdown("---")

    st.markdown("### 🧠 Example Questions")

    st.markdown("""
• What are the symptoms of diabetes  
• What causes hypertension  
• What is asthma  
• Explain migraine headaches
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):

        if st.session_state.current_chat:
            st.session_state.conversations[st.session_state.current_chat] = []

        st.session_state.messages = []
        st.rerun()

# -----------------------------
# CHAT DISPLAY
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="msg-row" style="justify-content:flex-end;">
                <div class="user-msg">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="msg-row">
                <div class="avatar">
                 <img src="data:image/png;base64,{logo}" width="60">
                </div>
                <div class="bot-msg">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# INPUT
# -----------------------------
query = st.chat_input("Ask Medibot a medical question...")

# -----------------------------
# RESPONSE LOGIC
# -----------------------------
if query:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Create chat title if new chat
    if st.session_state.current_chat is None:

        title = query[:30] + "..." if len(query) > 30 else query

        st.session_state.current_chat = title
        st.session_state.conversations[title] = st.session_state.messages

    # Get answer
    with st.spinner("Medibot is searching medical knowledge..."):
        response = get_answer(query)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Update conversation store
    st.session_state.conversations[st.session_state.current_chat] = st.session_state.messages

    st.rerun()