# ============================================
# AI VIRTUAL INTERVIEWER USING GROQ + RAG
# ============================================

# app.py

import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Virtual Interviewer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Virtual Interviewer using GROQ + RAG")

st.markdown("---")

# ============================================
# GROQ API KEY
# ============================================

groq_api_key = st.text_input(
    "Enter GROQ API Key",
    type="password"
)

if not groq_api_key:
    st.warning("Please enter GROQ API Key")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# ============================================
# FILE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# ============================================
# USER INTRODUCTION
# ============================================

user_intro = st.text_area(
    "Tell me about yourself",
    placeholder="""
Example:
I am a Python developer with experience in
Machine Learning, Generative AI, and RAG applications.
"""
)

# ============================================
# PDF PROCESSING FUNCTION
# ============================================

def process_resume(uploaded_pdf):

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

        tmp_file.write(uploaded_pdf.read())

        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB Vector Store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

# ============================================
# QUESTION GENERATION FUNCTION
# ============================================

def generate_questions(vectorstore, user_intro):

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Similarity Search
    relevant_docs = retriever.get_relevant_documents(user_intro)

    # Combine Context
    context = "\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    # Prompt Template
    template = """
    You are an expert AI interviewer.

    Candidate Resume Context:
    {context}

    Candidate Introduction:
    {user_intro}

    Generate:
    1. Three Easy Technical Questions
    2. Two Hard Technical Questions

    Questions should be:
    - Personalized
    - Technical
    - Based on candidate skills
    """

    prompt = PromptTemplate(
        input_variables=["context", "user_intro"],
        template=template
    )

    final_prompt = prompt.format(
        context=context,
        user_intro=user_intro
    )

    # GROQ LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7
    )

    # Generate Response
    response = llm.invoke(final_prompt)

    return response.content

# ============================================
# MAIN EXECUTION
# ============================================

if uploaded_file and user_intro:

    if st.button("Generate Interview Questions"):

        # Process Resume
        with st.spinner("Processing Resume..."):

            vectorstore = process_resume(uploaded_file)

        # Generate Questions
        with st.spinner("Generating Interview Questions..."):

            questions = generate_questions(
                vectorstore,
                user_intro
            )

        # Output
        st.success("Interview Questions Generated Successfully!")

        st.markdown("---")

        st.subheader("🧠 Generated Interview Questions")

        st.write(questions)

else:

    st.info("Upload Resume and Enter Introduction")