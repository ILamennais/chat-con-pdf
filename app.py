import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import tempfile

st.set_page_config(page_title="Chat con tu PDF", layout="wide")
st.title("📄 Chat con tu PDF")

openai_api_key = st.text_input("🔑 Tu API Key de OpenAI", type="password")

uploaded_file = st.file_uploader("📤 Sube tu archivo PDF", type="pdf")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    reader = PdfReader(tmp_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_texts(texts, embeddings)

    query = st.text_input("❓ Escribe tu pregunta sobre el documento:")
    if query:
        docs = vectorstore.similarity_search(query)
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.markdown("🧠 **Respuesta:**")
        st.info(response)
