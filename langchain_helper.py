# langchain_helper.py

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

import os
import whisper

load_dotenv()

embeddings = OpenAIEmbeddings()


def create_vector_db_from_memos(memo_files) -> FAISS:
    model = whisper.load_model("base")

    for audio_file in memo_files:
        result = model.transcribe(audio_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(result)

        db = FAISS.from_documents(docs, embeddings)
        return db



def get_response_from_query(db, query, openai_api_key, k=4):
    # text-danvinci can handle 4097 tokens

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

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

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response