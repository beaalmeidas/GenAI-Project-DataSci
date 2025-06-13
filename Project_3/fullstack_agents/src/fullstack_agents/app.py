# app.py
import streamlit as st
from fullstack_agents.main import FullstackAgents  # ajuste o import conforme seu projeto

st.title("Interface CrewAI")

user_input = st.text_input("Digite sua solicitação:")

inputs = {
    'pedido_usuario': user_input
}

if st.button("Executar"):
    resultado = FullstackAgents().crew().kickoff(inputs)  # adapte para chamar sua função principal
    st.write("Resultado:")
    st.write(resultado)