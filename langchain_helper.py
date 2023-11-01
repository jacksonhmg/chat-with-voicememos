# langchain_helper.py

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


import numpy as np
import io


import os
import tempfile
import whisper

import librosa
import numpy as np

from langchain.document_loaders import TextLoader

import tempfile
import shutil


load_dotenv()

embeddings = OpenAIEmbeddings()


# def create_vector_db_from_memos(memo_files) -> FAISS:
#     model = whisper.load_model("base")

#     for file in memo_files:
#         result = model.transcribe(file)
#         loader = TextLoader(file)
#         transcript = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         docs = text_splitter.split_documents(transcript)


#     db = FAISS.from_documents(docs, embeddings)
#     return db


def create_vector_db_from_memos2(memo_files):
    model = whisper.load_model("base")

    for uploaded_file in memo_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Write the content of the uploaded file to the temporary file
            shutil.copyfileobj(uploaded_file, temp_file)
            temp_file_path = temp_file.name

        # Transcribe using the path to the temporary file
        result = model.transcribe(temp_file_path)
        print(result["text"])

        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(result["text"])
            f.flush()
            
            loader = TextLoader(f.name)
            documents = loader.load()


        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)

        #faiss = FAISS.from_texts(result["text"], embeddings)


        # Optionally delete the temporary file if you don't need it anymore
        os.remove(temp_file_path)

    return db


def create_document(text):
    return {"page_content": text}



def get_response_from_query(db, query, openai_api_key, k=4):
    # text-danvinci can handle 4097 tokens

    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003", openai_api_key=openai_api_key)



    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about a user's transcribed voice memos. 
        
        Answer the following question: {question}
        By searching the following memos: {docs}
                
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    print("Query:", query)
    print("Docs:", docs_page_content)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response