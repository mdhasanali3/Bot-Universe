# import streamlit as st
import requests
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
# import config
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS

load_dotenv()
app = Flask(__name__)
CORS(app)


@app.route("/medichatbot",methods=['POST'])
def medi():
    print("i am in")
   
    # JSON data
    json_data = request.get_data("prompt","")

    # Parse the JSON data
    data = json.loads(json_data)

    # Extract the value of the "prompt" key
    prompt = data.get("prompt")
    print(prompt)
    medibot= mcb(prompt)
    print(medibot,"############medi")
    return jsonify({'bot': medibot})
    # Create a send button
    # if st.button("Send"):
     
       
    #     st.write("your answer :", ans)
def mcb(prompt):
    reader = PdfReader('./content/brochure.pdf')
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")
            # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

    text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 2000,
            chunk_overlap  = 100,
            length_function = len,
        )
    texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # query = "tell me about PERIOFLOW APPLICATIONS"
    query=prompt
    print("i am query ", query)
    docs = docsearch.similarity_search(query)
    ans=chain.run(input_documents=docs, question=query)

    return ans  

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1',port=5000)
    
