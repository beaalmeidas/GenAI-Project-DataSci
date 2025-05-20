import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings          # nomic-embed
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from html_templates import css, user_template, bot_template

# pega texto do pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
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

        pdf_docs = st.file_uploader("Faça upload dos seus arquivos e aperte em Processar", accept_multiple_files=True)

        if st.button("Processar"):
            with st.spinner("Processando"):
                # pega texto do pdf
                raw_text = get_pdf_text(pdf_docs)

                # pega chunks do texto
                text_chunks = get_text_chunks(raw_text)

                # cria "vectorstore" (FAISS)
                vectorstore = get_vectorstore(text_chunks)

                # cria a "conversation chain"
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()