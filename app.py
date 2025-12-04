import streamlit as st
from google import genai
from dotenv import load_dotenv
import os

######## SETUP GEMINI CREDENTIALS #######

load_dotenv()  # Load environment variables from .env file
api_token = os.getenv('api_key')

if not api_token:
    raise ValueError("No API token found. Check your .env file.")

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-2.0-flash",
# #     api_key=api_token
# #     )

client = genai.Client(api_key=api_token)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

######## SETUP STREAMLIT ########
st.header('Bank Deposit Subscription Prediction + LLM Explainer')
st.markdown("AI-based Subscription Predictor")
st.divider()

######## STREAMLIT INTERFACE ########
poutcome_success = st.sidebar.radio('Customer conversion in last outreach?',
                 ["Successfully Converted", "Not Successful"])
if poutcome_success == "Successfully Converted":
    input_poutcome_success = 1
else:
    input_poutcome_success = 0







