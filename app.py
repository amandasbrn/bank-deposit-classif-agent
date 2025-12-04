import streamlit as st
import joblib
import pandas as pd
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

######## CALL NECESSARY FILES ########

# pickle

model = joblib.load('adaboost_trained.pkl')
selected_feature = joblib.load('selected_feature.pkl')
job_edu_encoder = joblib.load('job_edu_encoder.pkl')

# csv
feat_imp = pd.read_csv('adaboost_feature_imp.csv')
data = pd.read_csv('data_model.csv')

######## SETUP STREAMLIT ########
st.header('Bank Deposit Subscription Prediction + LLM Explainer')
st.markdown("AI-based Subscription Predictor")
st.divider()

######## STREAMLIT INTERFACE ########

# poutcome_success
input_poutcome_success = st.sidebar.radio('Customer conversion in last outreach?',
                 ["Successfully Converted", "Not Successful"])
if input_poutcome_success == "Successfully Converted":
    poutcome_success = 1
else:
    poutcome_success = 0

# duration
duration = st.sidebar.number_input('How long did the last interaction last? (seconds)',
                                   value=None)
st.sidebar.caption(f'min:{data['duration'].min()}, max: {data['duration'].max()}')

# job-edu

job_edu_df = data[['job','education','education_encoded']].drop_duplicates(subset=['job','education'])

job = st.sidebar.selectbox('Occupation:', options=job_edu_df['job'].unique())
edu_options = job_edu_df.loc[job_edu_df['job'] == job, 'education'].unique()

edu = st.sidebar.selectbox(
    'Latest education:',
    options=sorted(edu_options)
)

job_edu_str = f"{job}_{edu}"
job_edu = job_edu_encoder.transform(
    pd.DataFrame({"job_edu": [job_edu_str]})
)["job_edu"].iloc[0]

# was_contacted_before
input_was_contacted_before = st.sidebar.radio('Was the customer contacted?',
                 ["Yes", "No"])
if input_was_contacted_before == "Yes":
    was_contacted_before = 1
else:
    was_contacted_before = 0

# contact_cellular
input_contact_cellular = st.sidebar.radio('Did the customer contacted on cellular?',
                 ["Yes", "No"])
if input_contact_cellular == "Yes":
    contact_cellular = 1
else:
    contact_cellular = 0

# balance
balance = st.sidebar.number_input("Customer's balance",
                                   value=None)
st.sidebar.caption(f'min:{data['balance'].min()}, max: {data['balance'].max()}')

# previous
previous = st.sidebar.number_input('How many times was the customer contacted in the previous campaign?',
                                   value=None)
st.sidebar.caption(f'min:{data['previous'].min()}, max: {data['previous'].max()}')

pred_button = st.sidebar.button("Get prediction", type="primary")

user_input = pd.DataFrame({'duration':[duration],
                           'poutcome_success':[poutcome_success],
                           'job_edu':[job_edu],
                           'was_contacted_before':[was_contacted_before],
                           'contact_cellular':[contact_cellular],
                           'balance':[balance],
                           'previous':[previous]
                           })

######## MODELING ########

if pred_button:
    y_pred = model.predict(user_input)[0]
    y_proba = model.predict_proba(user_input)[0][1]
    if y_pred == 1:
        st.write("This customer is likely to subscribe")
    else:
        st.write("This customer is unlikely to subscribe")