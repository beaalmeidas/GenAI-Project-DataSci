import streamlit as st
import os
import io

import database
from utils import extract_text_from_files, get_text_chunks

# UI de Criação de Conta/Login
def display_auth_ui():
    if "logged_in_user_id" not in st.session_state:
        st.session_state.logged_in_user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.logged_in_user_id is None:
        st.sidebar.subheader("Login ou Criar Conta")

        with st.sidebar.form(key="auth_form_sidebar"):
            username_input = st.text_input("Usuário", key="auth_username")
            password_input = st.text_input("Senha", type="password", key="auth_password")
            
            # Botões de Criar conta/Login
            col1, col2 = st.columns(2)
            login_button = col1.form_submit_button("Login")
            create_account_button = col2.form_submit_button("Criar Conta")

            # Botão Login
            if login_button:
                from werkzeug.security import check_password_hash

                user = database.get_user(username_input)
                
                if user and check_password_hash(user["password_hash"], password_input):
                    st.session_state.logged_in_user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.sidebar.success(f"Login como: {st.session_state.username}")
                    st.rerun()
                else:
                    st.sidebar.error("Usuário ou senha inválidos.")

            # Botão Criar Conta
            if create_account_button:
                if username_input and password_input:
                    existing_user = database.get_user(username_input)
                    if existing_user:
                        st.sidebar.warning("Usuário já existe.")
                    else:
                        from werkzeug.security import generate_password_hash
                        
                        hashed_password = generate_password_hash(password_input)
                        user_id = database.add_user(username_input, hashed_password)

                        if user_id:
                            st.sidebar.success("Conta criada! Faça o login.")
                        else:
                            st.sidebar.error("Erro ao criar conta.")
                else:
                    st.sidebar.warning("Preencha usuário e senha.")
    else:
        st.sidebar.subheader(f"Logado como: {st.session_state.username}")
        if st.sidebar.button("Logout", key="logout_button_sidebar"):
            for key in list(st.session_state.keys()):
                if key in ["logged_in_user_id", "username", "conversation", "chat_history", "vectorstore_loaded_for_user", "processed_files_session"]:
                    if key in st.session_state:
                        del st.session_state[key]
            st.sidebar.info("Logout realizado.")
            st.rerun()

# Processa pergunta do usuário, interagee com conversation_chain e FAISS
def handle_user_input(user_question, get_conversation_chain_func, save_chat_message_func):
    
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.warning("Por favor, processe alguns arquivos primeiro ou verifique se o conhecimento foi carregado.")
        return

    # conversation_chain chamado por (st.session_state.conversation)
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'] 

    if st.session_state.get("logged_in_user_id") and len(st.session_state.chat_history) >= 2:
        if hasattr(st.session_state.chat_history[-2], 'content') and hasattr(st.session_state.chat_history[-1], 'content'):
            last_user_msg = st.session_state.chat_history[-2].content
            last_ai_msg = st.session_state.chat_history[-1].content
            save_chat_message_func(st.session_state.logged_in_user_id, last_user_msg, last_ai_msg)

# UI para mostrar arquivos na sidebar
def display_uploaded_files_ui(handle_file_removal_func, faiss_index_name_const):
    
    user_id = st.session_state.get("logged_in_user_id")
    files_to_display = []           # lista de arquivos

    if user_id:         # se logado
        db_files = database.get_user_files(user_id)
        for db_file in db_files:
            files_to_display.append({'id': db_file['id'], 'name': db_file['filename'], 'source': 'db'})
    else:               # se deslogado (por sessão)
        for session_file in st.session_state.get("processed_files_session", []):
            files_to_display.append({'id': session_file['name'], 'name': session_file['name'], 'source': 'session'})

    if not files_to_display:
        st.sidebar.info("Nenhum arquivo carregado ainda.")
        return

    st.sidebar.subheader("Arquivos Carregados:")
    for file_info in files_to_display:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(file_info['name'])

        button_key = f"remove_{file_info['source']}_{str(file_info['id']).replace('.', '_').replace(' ', '_')}"
        
        # Remoção de arquivos
        if col2.button("🗑️", key=button_key, help=f"Remover {file_info['name']}"):
            handle_file_removal_func(file_info['id'], file_info['name'], file_info['source'], faiss_index_name_const)           # função auxiliar
            st.rerun()

# Função Auxiliar para lidar com lógica de remoção de arquivos
def handle_file_removal_logic(file_identifier, file_name_for_display, source, faiss_index_name_const, get_vectorstore_func, get_conversation_chain_func):
    
    user_id = st.session_state.get("logged_in_user_id")

    # Se usuário logado, infos -> DB
    if source == 'db' and user_id:
        if database.delete_user_file(file_identifier):
            st.sidebar.success(f"Arquivo '{file_name_for_display}' removido com sucesso!")
            
            user_faiss_dir_path = database.get_user_faiss_path(user_id)
            faiss_file_path = os.path.join(user_faiss_dir_path, f"{faiss_index_name_const}.faiss")          # índice .faiss
            pkl_file_path = os.path.join(user_faiss_dir_path, f"{faiss_index_name_const}.pkl")              # índice .pkl
            
            if os.path.exists(faiss_file_path): os.remove(faiss_file_path)
            if os.path.exists(pkl_file_path): os.remove(pkl_file_path)
            
            st.warning("Seu banco de conhecimento foi zerado. Processe os arquivos desejados novamente!")
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.vectorstore_loaded_for_user = False
        else:
            st.sidebar.error(f"Erro ao remover '{file_name_for_display}' do registro.")

    # Se usuário deslogado, infos -> sessão
    elif source == 'session' and not user_id:
        st.session_state.processed_files_session = [
            f for f in st.session_state.processed_files_session if f['name'] != file_identifier
        ]
        st.sidebar.success(f"Arquivo '{file_name_for_display}' removido da sessão.")

        if not st.session_state.processed_files_session:
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.info("Todos os arquivos da sessão foram removidos!")
        else:
            with st.spinner("Atualizando conhecimento da sessão..."):
                mock_uploaded_files = []
                for file_data in st.session_state.processed_files_session:
                    bytes_io_obj = io.BytesIO(file_data['bytes'])
                    bytes_io_obj.name = file_data['name'] 
                    mock_uploaded_files.append(bytes_io_obj)

                raw_text = extract_text_from_files(mock_uploaded_files) 
                if raw_text and raw_text.strip():
                    text_chunks = get_text_chunks(raw_text) 
                    vectorstore = get_vectorstore_func(text_chunks=text_chunks, user_id=None)
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain_func(vectorstore, initial_chat_history=[]) 
                        st.session_state.chat_history = []
                        st.success("Conhecimento da sessão atualizado.")
                    else:
                        st.session_state.conversation = None; st.session_state.chat_history = []
                        st.error("Falha ao reconstruir conhecimento da sessão.")
                else:
                    st.session_state.conversation = None; st.session_state.chat_history = []
                    st.info("Nenhum texto para processar após remoção.")
    else:
        st.error("Erro: Estado inconsistente ao tentar remover arquivo.")