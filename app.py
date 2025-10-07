import streamlit as st
from dotenv import load_dotenv
import os

# SETUP GEMINI CREDENTIALS

load_dotenv()  # Load environment variables from .env file
api_token = os.getenv('api_key')

# if not api_token:
#     raise ValueError("No API token found. Check your .env file.")

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.0-flash",
# #     api_key=api_token
# #     )

# SETUP STREAMLIT
st.header('Test')