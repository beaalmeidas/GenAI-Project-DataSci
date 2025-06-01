import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
import csv
import io
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# (.pdf; .docx; .xlsx; .csv; .txt; .md)
def extract_text_from_files(uploaded_files):
    text = ""
    for file in uploaded_files:
        filename = file.name.lower()
        try:
            if filename.endswith(".pdf"):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif filename.endswith(".docx"):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif filename.endswith(".xlsx"):
                wb = openpyxl.load_workbook(file, data_only=True)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        text += ' '.join([str(cell) if cell is not None else '' for cell in row]) + "\n"
            elif filename.endswith(".csv"):
                try:
                    decoded = file.read().decode("utf-8")           # .csv (utf-8)
                except UnicodeDecodeError:
                    file.seek(0) 
                    try:
                        decoded = file.read().decode("latin-1")     # .csv (latin-1)
                    except UnicodeDecodeError:
                        file.seek(0)
                        decoded = file.read().decode("utf-8", errors='replace')
                reader = csv.reader(io.StringIO(decoded))
                for row in reader:
                    text += ' | '.join(row) + "\n"
            elif filename.endswith(".txt") or filename.endswith(".md"):
                try:
                    text += file.read().decode("utf-8") + "\n"                  # .txt/.md (utf-8)
                except UnicodeDecodeError:
                    file.seek(0)
                    try:
                        text += file.read().decode("latin-1") + "\n"            # .txt/.md (latin-1)
                    except UnicodeDecodeError:
                        file.seek(0)
                        text += file.read().decode("utf-8", errors='replace') + "\n"
            else:
                st.warning(f"Formato de arquivo não suportado: {file.name}")
                text += f"\n[Unsupported file format: {file.name}]\n"
        except Exception as e:
            st.error(f"Erro ao processar o arquivo {file.name}: {e}")
            text += f"\n[Error processing file: {file.name}]\n"
    return text

# Chunking
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore, initial_chat_history=None):
    llm = ChatOllama(model="llama3", temperature=0.1)                       # uso do llama3 (use com -> ollama serve)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    # Caso exista histórico de conversa
    if initial_chat_history: 
        memory.chat_memory.messages = initial_chat_history

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
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory, 
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return conversation_chain

# Vectorstore FAISS
def get_vectorstore(text_chunks=None, user_id=None, db_get_user_faiss_path_func=None, faiss_index_name_const=None, session_state=None, st_feedback_obj=None):
    
    # Embeddings através do nomic (Hugging Face API)
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
    
    if user_id: 
        if not all([db_get_user_faiss_path_func, faiss_index_name_const, session_state, st_feedback_obj]):
            if st_feedback_obj:
                st_feedback_obj.error("Erro Interno! Tente Novamente mais tarde.")
            return None

        user_faiss_dir_path = db_get_user_faiss_path_func(user_id)
        faiss_index_file_path = os.path.join(user_faiss_dir_path, f"{faiss_index_name_const}.faiss")

        if text_chunks: 
            if not os.path.exists(user_faiss_dir_path):
                os.makedirs(user_faiss_dir_path)
            
            vectorstore = None          # Inicialização do Vectorstore

            if os.path.exists(faiss_index_file_path):
                try:        # Se já existir vectorstore do usuário, é carregado
                    local_vectorstore = FAISS.load_local(user_faiss_dir_path, embeddings, faiss_index_name_const, allow_dangerous_deserialization=True)
                    local_vectorstore.add_texts(texts=text_chunks) 
                    local_vectorstore.save_local(user_faiss_dir_path, faiss_index_name_const)
                    vectorstore = local_vectorstore
                except Exception as e:
                    st_feedback_obj.error(f"Erro ao atualizar índice FAISS: {e}. Criando um novo índice.")
                    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    vectorstore.save_local(user_faiss_dir_path, faiss_index_name_const)
            else:           # se não existir, cria um novo
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                vectorstore.save_local(user_faiss_dir_path, faiss_index_name_const)
            return vectorstore
        
        else: # Nenhum chunk de texto, apenas carrega
            if os.path.exists(faiss_index_file_path):
                try:
                    vectorstore = FAISS.load_local(user_faiss_dir_path, embeddings, faiss_index_name_const, allow_dangerous_deserialization=True)
                    session_state.vectorstore_loaded_for_user = True 
                    return vectorstore
                except Exception as e:
                    st_feedback_obj.error(f"Erro ao carregar índice FAISS: {e}. Faça upload de arquivos para criar um novo.")
                    session_state.vectorstore_loaded_for_user = False 
                    return None
            else:
                session_state.vectorstore_loaded_for_user = False
                return None 
                
    else: # Usuário não logado (Fluxo padrão)
        if text_chunks:
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        return None