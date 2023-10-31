import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        api_key = st.sidebar.text_input(
            label="What is your OpenAI API key?",
            max_chars=200,
            key="api_key",
        )
        memo_files = st.file_uploader(
            label="Upload all your memo files",
            accept_multiple_files=True,
        )

        submit_button = st.form_submit_button(label='Submit')

if memo_files and api_key:
    db = lch.create_vector_db_from_memos(memo_files)
    response = lch.get_response_from_query(db, query, api_key)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, 80))
        