import streamlit as st
import os
from dotenv import load_dotenv
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.messages import trim_messages

# 🌱 Load .env
load_dotenv()

# 🧠 Embeddings + Retriever
embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key=os.getenv("HUGGING_FACE")
)
retriever = FAISS.load_local('VectorDB', embeddings, allow_dangerous_deserialization=True).as_retriever()

# 💬 Model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=os.getenv("GROQ_API_KEY"))

# 🔁 Chat memory store
if "store" not in st.session_state:
    st.session_state.store = {}

def chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 📜 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful and insightful assistant — a digital extension of Hemanth Puppala himself. Speak in a clear, confident, and professional tone that reflects Hemanth’s style: thoughtful, technically sharp, and approachable. Your responses should be concise but not curt — detailed enough to be genuinely helpful, yet never overwhelming.
If the question is about Hemanth Puppala, respond **only** with context below.
<context>
{context}
</context>
Provide a response that is concise yet informative. Avoid excessive elaboration, but ensure the key points are fully addressed.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 🔗 Prompt | Model
core_chain = prompt | llm

# 🧠 Memory wrapper
with_memory = RunnableWithMessageHistory(
    core_chain,
    get_session_history=chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 📚 Retrieval + memory injection
retrieval_chain = RunnableMap({
    "input": lambda x: x["input"],
    "context": lambda x: retriever.invoke(x["input"]),
    "session_id": lambda x: x["session_id"]
})

inject_memory = RunnableLambda(
    lambda inputs: {
        **inputs,
        "chat_history": trim_messages(
            chat_history(inputs["session_id"]).messages,
            max_tokens=1000,
            token_counter = llm
        )
    }
)

# 🔁 Full chain
full_chain = retrieval_chain | inject_memory | with_memory

# 🧠 Session ID (could be user ID)

# 🚀 Streamlit UI
st.title("Hemanth Puppala's AI Assistant")
st.markdown(
    "<span style='font-size: 12px; color: #999999;'>This assistant might occasionally produce incorrect or outdated information</span>",
    unsafe_allow_html=True
)
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = ["chat1"]
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "chat1"

with st.sidebar:
    st.title("Chat Sessions")

    # Add new chat session
    if st.button("➕ New Chat"):
        new_chat_id = f"chat{len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions.append(new_chat_id)
        st.session_state.current_chat = new_chat_id
        st.rerun()  # reload to reflect change

    # Select existing chat
    selected = st.radio(
        "Select a chat",
        st.session_state.chat_sessions,
        index=st.session_state.chat_sessions.index(st.session_state.current_chat)
    )

    # Update current session ID
    st.session_state.current_chat = selected
    

SESSION_ID = st.session_state.current_chat


if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}

if SESSION_ID not in st.session_state.store:
    st.session_state.store[SESSION_ID] = ChatMessageHistory()

history = st.session_state.store[SESSION_ID].messages

# Sync display messages
st.session_state.messages = [
    {"role": "user" if m.type == "human" else "AI", "content": m.content}
    for m in history
]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input box
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("AI"), st.spinner("Thinking..."):
        result = full_chain.invoke(
            {"input": user_input, "session_id": SESSION_ID},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        answer = result.content if hasattr(result, "content") else str(result)
        answer = re.sub(r"<think>.*?</think>", "", answer).strip()
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})
    # st.markdown("**Chat Memory Log**")
    # for msg in chat_history(SESSION_ID).messages:
    #     st.markdown(f"- **{msg.type.upper()}**: {msg.content}")
    # add a light disclaimer at bottom left
    
