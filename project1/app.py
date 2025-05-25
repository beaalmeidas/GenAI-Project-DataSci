import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
import csv
import io

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings          # nomic-embed
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from html_templates import css, user_template, bot_template

# Realiza extração de textos de diferentes tipos de arquivos
def extract_text_from_files(uploaded_files):
    text = ""
    for file in uploaded_files:
        filename = file.name.lower()

        # Extração de .pdf
        if filename.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        # Extração de .docx
        elif filename.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        # Extração de .xlsx
        elif filename.endswith(".xlsx"):
            wb = openpyxl.load_workbook(file, data_only=True)
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    text += ' '.join([str(cell) if cell is not None else '' for cell in row]) + "\n"

        # Extração de .csv
        elif filename.endswith(".csv"):
            decoded = file.read().decode("utf-8")
            reader = csv.reader(io.StringIO(decoded))
            for row in reader:
                text += ' | '.join(row) + "\n"

        # Extração de .txt ou .md
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text += file.read().decode("utf-8") + "\n"

        else:
            text += f"\n[Unsupported file format: {file.name}]\n"
    
    return text


# pega chunks do texto
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


# cria "vectorstore" (FAISS)
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})         # usando o nomic-embed-text-v1
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


# cria a "conversation chain"
def get_conversation_chain(vectorstore):

    llm = ChatOllama(model="llama3", temperature=0.1)           # usando o llama3

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)          # memória histórico de conversa

    # Prompt Template
    CUSTOM_PROMPT_TEMPLATE = """
    Você é um assistente de IA altamente especializado em responder perguntas com base em documentos fornecidos pelo usuário.
    Seu objetivo é fornecer respostas **precisas**, **claras** e **baseadas estritamente nas informações** extraídas dos arquivos carregados.
    Se a resposta não puder ser encontrada nos documentos, **admita que não sabe** ao invés de inventar uma resposta.
    Você pode usar informações de múltiplos documentos. Se as informações não estiverem diretamente conectadas, diga isso claramente.
    Responda SEMPRE em **português do Brasil**, mesmo que os documentos ou a pergunta estejam em outro idioma.
    ---
    Histórico da conversa:
    {chat_history}

    Pergunta do usuário:
    {question}

    Contexto dos documentos:
    {context}

    Resposta:
    """

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question", "context"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, combine_docs_chain_kwargs={"prompt": prompt})
    return conversation_chain



def handle_userInput(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

 
def main():
    load_dotenv()

    st.set_page_config(page_title="RAG - Múltiplos PDF's", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)

    st.header("RAG - Múltiplos PDF's :books:")
    user_question = st.text_input("Faça uma pergunta sobre os seus arquivos: ")
    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("Seu arquivos:")

        pdf_docs = st.file_uploader(
            "Faça upload dos seus arquivos e aperte em Processar (.pdf, .docx, .xlsx, .csv, .txt, .md)",
            accept_multiple_files=True,
            type=["pdf", "docx", "xlsx", "csv", "txt", "md"]
        )

        if st.button("Processar"):
            with st.spinner("Processando"):
                # pega texto do pdf
                raw_text = extract_text_from_files(pdf_docs)

                # pega chunks do texto
                text_chunks = get_text_chunks(raw_text)

                # cria "vectorstore" (FAISS)
                vectorstore = get_vectorstore(text_chunks)

                # cria a "conversation chain"
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()