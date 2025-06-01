from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

app = Flask(__name__)

@app.route("/")
def health():
    return "Insurance PDF GPT Agent is running."

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["pdf"]
    question = request.form["question"]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file.save(temp_file.name)

    doc = fitz.open(temp_file.name)
    text = "\n".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    db = FAISS.from_texts(chunks, OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        retriever=db.as_retriever()
    )
    answer = qa.run(question)

    return jsonify({"answer": answer})
