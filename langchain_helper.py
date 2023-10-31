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

embeddings = OpenAIEmbeddings()


def create_vector_db_from_memos(memo_files) -> FAISS:
    model = whisper.load_model("base")

    for audio_file in memo_files:
        result = model.transcribe(audio_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(result)

        db = FAISS.from_documents(docs, embeddings)
        return db
        # Do something with the embedding
