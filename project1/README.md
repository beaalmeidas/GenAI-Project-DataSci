# Chat RAG com Login e Persistência de Dados

Um sistema de chat inteligente construído com Streamlit e Langchain que permite aos usuários realizar upload de documentos e realizar perguntas sobre o conteúdo, com suporte a contas de usuário, histórico de conversas e persistência de arquivos.

## Visão Geral

Este projeto implementa o padrão RAG (Retrieval-Augmented Generation) para criar um chatbot que responde perguntas com base em um conjunto de documentos fornecido pelo usuário. A aplicação é totalmente interativa, com uma interface web construída em Streamlit, e possui um sistema de backend robusto para gerenciar usuários, conversas e os índices vetoriais de cada um.

---

## ✨ Funcionalidades Principais

* **Autenticação de Usuário:** Sistema de criação de conta e login para garantir que os dados de cada usuário sejam privados e persistentes.
* **Upload de Múltiplos Formatos:** Suporte para upload de arquivos `.pdf`, `.docx`, `.xlsx`, `.csv`, e `.txt`.
* **Persistência de Dados por Usuário:**
    * **Histórico de Chat:** As conversas são salvas em um banco de dados SQLite e recarregadas a cada login.
    * **Base de Conhecimento:** Os documentos processados são convertidos em vetores e armazenados em um índice FAISS dedicado para cada usuário.
* **Gerenciamento de Arquivos:** Os usuários podem visualizar e remover os arquivos que já foram carregados, com a base de conhecimento sendo atualizada de acordo.

---

## 🛠️ Tecnologias Utilizadas

* **[Streamlit](https://streamlit.io/)**
* **Python 3.12+**
* **LLM e RAG:** [Ollama](https://ollama.com/) com o modelo `llama3` e [Langchain](https://www.langchain.com/).
* **Banco de Dados Relacional:** [SQLite](https://www.sqlite.org/)
* **Banco de Dados Vetorial:** [Facebook AI Similarity Search (FAISS)](https://github.com/facebookresearch/faiss)
* **Modelo de Embeddings:** `nomic-ai/nomic-embed-text-v1` da [Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1)

---

## 🚀 Instalação e Execução

Siga os passos abaixo para configurar e rodar o projeto localmente.

### 1. Pré-requisitos

* **Python 3.12 ou superior:** [Instalar Python](https://www.python.org/downloads/)
* **Git:** Para clonar o repositório.
* **Ollama:** A aplicação utiliza o Ollama para rodar o modelo Llama 3 localmente.
    * [Instale o Ollama](https://ollama.com/).
    * Após a instalação, puxe o modelo `llama3` com o comando:
        ```bash
        ollama pull llama3
        ```
    * Certifique-se de que o Ollama esteja em execução antes de iniciar a aplicação Streamlit.

### 2. Clone o Repositório

```bash
git clone https://github.com/beaalmeidas/GenAI-Project-DataSci
cd project1
```

### 3. Crie um Ambiente Virtual e Instale as Dependências
```bash
# Criar o ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
venv\Scripts\activate           # <- Windows

source venv/bin/activate        # <- macOS/Linux
```

Agora, instale todas as dependências com:

```bash
pip install -r requirements.txt
```

## 4. Execute a Aplicação
Com o ambiente virtual ativado, execute o Ollama e inicie a aplicação Streamlit:

<h2> Iniciar llama3 </h2>

```bash
ollama serve
```

<h2> Iniciar aplicação Streamlit</h2>

```bash
streamlit run app.py
```

A aplicação deverá abrir automaticamente no seu navegador padrão.


## 📖 Como Usar

1. **Crie uma Conta:** Na barra lateral, insira um nome de usuário e senha e clique em "Criar Conta".


2. **Faça Login:** Use as mesmas credenciais para fazer o login.


3. **Faça Upload de Arquivos:** Na barra lateral, selecione um ou mais documentos e clique no botão "Processar Arquivos".

4. **Aguarde o Processamento:** A aplicação irá extrair o texto, criar os embeddings e salvar sua base de conhecimento.

5. **Converse:** Com os arquivos processados, vá para a área de chat principal, digite sua pergunta e clique em "Enviar".

6. **Gerencie seus Arquivos:** Na barra lateral, você pode ver a lista de arquivos carregados e remover qualquer um deles clicando no ícone de lixeira.
