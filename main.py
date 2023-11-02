# main.py

import streamlit as st
import langchain_helper as lch
import textwrap



from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
import os
import MySQLdb

# Connect to the database
connection = MySQLdb.connect(
  host=os.getenv("DATABASE_HOST"),
  user=os.getenv("DATABASE_USERNAME"),
  passwd=os.getenv("DATABASE_PASSWORD"),
  db=os.getenv("DATABASE"),
  autocommit=True,
  ssl_mode="VERIFY_IDENTITY",
  # See https://planetscale.com/docs/concepts/secure-connections#ca-root-configuration
  # to determine the path to your operating systems certificate file.
  ssl={ "ca": "/etc/ssl/cert.pem" }
)

try:
    # Create a cursor to interact with the database
    cursor = connection.cursor()

    # Execute "SHOW TABLES" query
    cursor.execute("SHOW TABLES")

    # Fetch all the rows
    tables = cursor.fetchall()

    # Print out the tables
    print("Tables in the database:")
    for table in tables:
        print(table[0])

except MySQLdb.Error as e:
    print("MySQL Error:", e)

finally:
    # Close the cursor and connection
    cursor.close()
    connection.close()



















st.title("Chat With Voice Memos!")
st.write("Airdrop your voice memos from your phone to macbook and upload them here")

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
            key="memo_files",
        )
        query = st.sidebar.text_area(
            label="Ask a question about your memos",
            max_chars=300,
            key="query",
        )
        submit_button = st.form_submit_button(label='Submit')

# Check if the database is already created in the session state
if 'db' not in st.session_state:
    st.session_state.db = None

# Create the database only if memo files are uploaded and the database does not exist
if memo_files and api_key and not st.session_state.db:
    st.session_state.db = lch.create_vector_db_from_memos2(memo_files, api_key, st)

# Query submission and response
if query and st.session_state.db:
    response = lch.get_response_from_query(st.session_state.db, query, api_key, st)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, 80))
