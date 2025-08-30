from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from htmlTemplates import css, bot_template, user_template

# ----------------------------
# Custom Prompt for Medical Research Assistant
# ----------------------------
custom_template = """You are a Medical Research Assistant. 
You help summarize and explain insights from medical research papers, 
clinical trial reports, and scientific articles. Always use the provided 
context and cite sources with [filename, page/line]. 
If the answer is uncertain, say 'Not enough information found in the documents.'

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# ----------------------------
# File Processing (PDF + TXT + DOCX)
# ----------------------------
def get_file_text(docs):
    texts = []
    for file in docs:
        suffix = os.path.splitext(file.name)[1].lower()

        # Save uploaded file to a temp file (needed for loaders)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for i, page in enumerate(pages):
                if page.page_content.strip():
                    texts.append({
                        "content": page.page_content,
                        "metadata": {"source": file.name, "page": i + 1}
                    })

        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            loaded_docs = loader.load()
            for i, d in enumerate(loaded_docs):
                texts.append({
                    "content": d.page_content,
                    "metadata": {"source": file.name, "line": i + 1}
                })

        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
            loaded_docs = loader.load()
            for i, d in enumerate(loaded_docs):
                texts.append({
                    "content": d.page_content,
                    "metadata": {"source": file.name, "section": i + 1}
                })

    return texts

# ----------------------------
# Chunking
# ----------------------------
def get_chunks(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = []
    for item in raw_texts:
        for chunk in text_splitter.split_text(item["content"]):
            chunks.append({
                "content": chunk,
                "metadata": item["metadata"]
            })
    return chunks

# ----------------------------
# Build FAISS Vectorstore
# ----------------------------
def get_vectorstore(chunks):
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    from langchain.docstore.document import Document
    documents = [Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks]
    # FAISS.from_documents expects the embeddings as a second positional argument
    return FAISS.from_documents(documents, embeddings)

# ----------------------------
# Conversation Chain
# ----------------------------
def get_conversationchain(vectorstore):
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("⚠️ OPENROUTER_API_KEY not found. Please add it to your .env file.")
        return None

    # Use OpenRouter as an OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",  # can be changed
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True
    )


# ----------------------------
# Handle Q&A
# ----------------------------
def handle_question(question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process documents first.")
        return

    response = st.session_state.conversation({"question": question})

    # 1) Prefer messages from the chain memory (if available)
    messages = []
    conv = st.session_state.conversation
    mem = getattr(conv, "memory", None)
    if mem:
        chat_mem = getattr(mem, "chat_memory", None)
        if chat_mem:
            messages = getattr(chat_mem, "messages", []) or []

    # 2) Fallback to chat_history returned in response
    if not messages:
        messages = response.get("chat_history", []) or []

    # 3) If no structured messages exist, only show the chain's answer (single bubble)
    if not messages:
        answer = response.get("answer") or response.get("result") or response.get("output")
        if answer:
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
        # show sources if any
    else:
        # 4) Render messages, but combine consecutive assistant messages into one bubble
        out_blocks = []  # list of tuples (role, content)
        prev_role = None
        buffer = []

        def flush_buffer():
            if not buffer:
                return
            combined = "\n\n".join([b.strip() for b in buffer if b and isinstance(b, str)])
            out_blocks.append((prev_role or "assistant", combined))
            buffer.clear()

        for msg in messages:
            # support Message objects and dict-like messages
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "assistant")
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            role = "user" if role in ("human", "user") else "assistant"

            if prev_role is None:
                prev_role = role

            if role == prev_role:
                buffer.append(content or "")
            else:
                flush_buffer()
                prev_role = role
                buffer.append(content or "")

        flush_buffer()

        # 5) Display the consolidated blocks (one bubble per user/assistant block)
        for role, content in out_blocks:
            if role == "user":
                st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

    # 6) Show unique citations (dedupe by source + page/line/section)
    src_docs = response.get("source_documents", []) or []
    if src_docs:
        seen_sources = set()
        st.subheader("📖 Sources")
        for doc in src_docs:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source", "unknown")
            # try page, then line, then section, else blank
            loc = meta.get("page") or meta.get("line") or meta.get("section") or ""
            key = (src, str(loc))
            if key in seen_sources:
                continue
            seen_sources.add(key)

            if "page" in meta:
                st.markdown(f"- {src} (Page {meta.get('page','?')})")
            elif "line" in meta:
                st.markdown(f"- {src} (Line {meta.get('line','?')})")
            elif "section" in meta:
                st.markdown(f"- {src} (Section {meta.get('section','?')})")
            else:
                st.markdown(f"- {src}")
# ----------------------------
# Main App
# ----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Medical Research Assistant", page_icon="🩺")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("🩺 Chat with Medical Research Papers, Notes & Reports")

    question = st.text_input("Ask a question from your documents:")
    if question and st.session_state.conversation:
        handle_question(question)
    elif question and not st.session_state.conversation:
        st.warning("⚠️ Please upload and process research papers/notes first.")

    with st.sidebar:
        st.subheader("📂 Upload Research Papers or Notes")
        docs = st.file_uploader(
            "Upload PDFs, TXT, or DOCX files and click 'Process'",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )
        if st.button("Process") and docs:
            with st.spinner("Processing..."):
                raw_texts = get_file_text(docs)
                text_chunks = get_chunks(raw_texts)
                if not text_chunks:
                    st.error("❌ No text extracted. Try another file.")
                else:
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversationchain(vectorstore)

        st.sidebar.info("""
        🔹 This assistant is specialized for **Medical Research Papers & Notes**.  
        Example queries:
        - "What are the side effects of the drug studied in this trial?"  
        - "Summarize findings on COVID-19 treatments."    
        """)



 
if __name__ == '__main__':
    main()