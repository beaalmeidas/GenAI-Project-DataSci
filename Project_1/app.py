from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

import os
import io
import subprocess
from docx import Document
import pdfplumber
import tempfile


def convert_to_markdown(uploaded_file):
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".md", ".txt"]:
        content = uploaded_file.read()
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

    elif ext == ".pdf":
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text

    elif ext == ".docx":
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")
    
        
def extract_chunks(content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(content)
    return chunks

def vector_database(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def query_vector_database(vectorstore):
    llm = Ollama(model="llama3", temperature=0.1)
    
    CUSTOM_PROMPT_TEMPLATE = """
    Você é um assistente útil que responde **exclusivamente em Português Brasileiro (pt-BR)**.
    Use um tom formal e claro.

    Contexto: {context}
    Pergunta: {question}

    Responda em Português:
    """
    
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"] 
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt} 
    )
    
    return rag_chain


import streamlit as st

st.set_page_config(page_title="Chat Assistente ~ RAG", layout="centered")

col1, col2 = st.columns(2)

with col1:
    st.title("📄 Chat com PDF usando RAG e IA Generativa")
    st.write("""
            1. Langchain
            2. Nomic (Ollama)
            3. Llama3 (Ollama)
            4. Streamlit""")

    # Inicializa variáveis no session_state se ainda não existirem
    if "content" not in st.session_state:
        st.session_state.content = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


    uploaded_file = st.file_uploader("Faça o upload de ao menos um arquivo", 
                                    type=[
                                        ".txt",
                                        ".pdf",
                                        ".docx",
                                        ".md",
                                        ]
                                    )

    if uploaded_file:
        if st.session_state.content is None:
            try:
                st.session_state.content = convert_to_markdown(uploaded_file)
                st.success("Arquivo convertido com sucesso!")

                with st.spinner("Processando..."):
                    chunks = extract_chunks(st.session_state.content)
                    vectorstore = vector_database(chunks)
                    st.write("Banco de dados vetorial criado com sucesso!")
                    st.session_state.rag_chain = query_vector_database(vectorstore)

            except Exception as e:
                st.error(f"Erro ao converter o arquivo: {str(e)}")
                st.stop()
                
    with col2:
        st.write("##")  
        st.write("##")  
        st.subheader("Chat com PDF usando RAG e IA Generativa")
        user_input = st.text_input("Digite sua pergunta:")

        if user_input and st.session_state.rag_chain:
            with st.spinner("Buscando resposta..."):
                response = st.session_state.rag_chain.run(user_input)
                st.write("Resposta encontrada:")
                st.write(response)
        else:
            st.warning("Por favor, faça o upload de um arquivo para continuar.")
            st.stop()