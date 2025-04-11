import streamlit as st
from langchain_community.document_loaders import  TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_chroma import Chroma
from langchain.llms import Ollama
import os
import re
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')
os.environ['LANGSMITH_TRACING'] = "true"
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
embeddings = OllamaEmbeddings(model='mxbai-embed-large')
chromaa = FAISS.load_local('VectorDB', embeddings,allow_dangerous_deserialization=True)
llm = Ollama(model="qwen2.5-coder")

#prompt using {question} to match ConversationalRetrievalChain
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are Hemanth Puppala's AI assistant — engineered by Hemanth himself. 
You're here to help users with anything they need: answering questions, solving problems.

Speak in a professional tone — like a helpful professional assistant.

If the question is about Hemanth Puppala — his work, background, or anything personal/professional — respond **strictly using the context below**. Otherwise, feel free to respond naturally and conversationally.
discourage redundant greetings, unless it's clearly the start of a conversation.
<context>
{context}
</context>

Be concise, clear, and witty where appropriate. You're allowed to have personality — just don't make stuff up.
"""
    ),
    ("human", "{question}") 
])

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# retriever
retriever = chromaa.as_retriever()

# conversational retrieval chain with memory
from langchain.chains import ConversationalRetrievalChain

# conversational chain with memory
if "ret_chain" not in st.session_state:
    st.session_state.ret_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},  
    )
print(st.session_state.ret_chain.input_keys)

# Streamlit UI
st.title("🧠 Test Chat with LLM")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#  input
inp = st.chat_input("Enter your message...")

if inp:
    # display user message
    st.session_state.messages.append({"role": "user", "content": inp})
    with st.chat_message("user"):
        st.markdown(inp)

    #  LLM response
    with st.spinner("Thinking..."):
        # Invoke the conversational chain with the correct input key "question"
        answerr = st.session_state.ret_chain.invoke({"question": inp})
        response = answerr["answer"]
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

    # ai message
    with st.chat_message("AI"):
        st.markdown(response)
    st.session_state.messages.append({"role": "AI", "content": response})
