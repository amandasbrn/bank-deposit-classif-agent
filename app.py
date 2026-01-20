import streamlit as st
import joblib
import json
import pandas as pd
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

######## SETUP GEMINI CREDENTIALS #######

load_dotenv()  # Load environment variables from .env file
api_token = os.getenv('api_key')

if not api_token:
    raise ValueError("No API token found. Check your .env file.")

# client = genai.Client(api_key=api_token)

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Explain how AI works in a few words",
# )

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

######## ML PREDICTION ########

# if pred_button:
#     y_pred = model.predict(user_input)[0]
#     y_proba = model.predict_proba(user_input)[0][1]
#     if y_pred == 1:
#         st.write("This customer is likely to subscribe")
#     else:
#         st.write("This customer is unlikely to subscribe")

######## LLM EXPLAINER ########
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_token
    )

system_instruction = """
You are an AI assistant helping a bank’s marketing team understand a predictive system for term deposit subscription.

The outreach action has ALREADY been selected by a decision agent.
You must NOT change or suggest a different action.

Your job is to:
- Explain in simple, non-technical language why this action was chosen
- Support the agent’s decision using customer context and model output
- Provide 3–5 concrete execution tips that align with the chosen action

Be concise, structured, and avoid exposing internal model details.
"""

top_features = [
    {
        "name": row["feature"],
        "importance": float(row["importance"])
    }
    for _, row in feat_imp.iterrows()
]

# === DECISION AGENT (RULE-BASED) ===
def marketing_decision_agent(probability, poutcome_success, duration, was_contacted_before):
    # Default decision
    decision = {
        "action": "do_not_contact",
        "priority": "low",
        "follow_up_window": None,
        "reason": "low_probability"
    }

    # High confidence customers
    if probability >= 0.6:
        decision["action"] = "call"
        decision["priority"] = "high"
        decision["follow_up_window"] = "24-48 hours"
        decision["reason"] = "high_subscription_probability"

    # Medium confidence
    elif 0.4 <= probability < 0.6:
        decision["action"] = "email"
        decision["priority"] = "medium"
        decision["follow_up_window"] = "2-3 days"
        decision["reason"] = "medium_subscription_probability"

    # Escalation rules
    if poutcome_success == 1:
        decision["priority"] = "high"
        decision["reason"] += "_previous_success"

    if duration > 300:
        decision["reason"] += "_high_engagement"

    return decision

# === PREDICTION + LLM FLOW ===
if pred_button:
    # ---- ML prediction ----
    y_pred = model.predict(user_input)[0]
    y_proba = float(model.predict_proba(user_input)[0][1])

    if y_pred == 1:
        st.success(f"This customer is likely to subscribe. Probability: {y_proba:.2f}")
    else:
        st.warning(f"This customer is unlikely to subscribe. Probability: {y_proba:.2f}")
    
    agent_decision = marketing_decision_agent(
    probability=y_proba,
    poutcome_success=poutcome_success,
    duration=duration,
    was_contacted_before=was_contacted_before)

    # ---- Build ml_output payload for LLM ----
    ml_output = {
        "prediction": int(y_pred),  # 0 or 1
        "prediction_label": "subscribe" if y_pred == 1 else "not_subscribe",
        "probability": y_proba,
        "agent_decision": agent_decision,
        "top_features": top_features,

        "input_features": {
            "poutcome_success": int(poutcome_success),
            "duration": float(duration),
            "job_edu": float(job_edu),  # already encoded
            "was_contacted_before": int(was_contacted_before),
            "contact_cellular": int(contact_cellular),
            "balance": float(balance),
            "previous": int(previous),
            # If you have campaign as a separate feature, add:
            # "campaign": int(campaign),
        },
    }

    ml_output_json = json.dumps(ml_output, ensure_ascii=False)
    
    # ---- LLM explainer ----
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_token,
        temperature=0.3
    )

    messages = [
        ("system", system_instruction),
        (
            "human",
            "Below is the output from a predictive system and a decision agent.\n\n"
            f"{ml_output_json}\n\n"
            "Explain why the agent selected this action and provide execution guidance "
            "that strictly follows the agent’s decision."
        ),
    ]


    ai_msg = llm.invoke(messages)

    st.subheader("Agent Decision")
    st.write(f"**Action:** {agent_decision['action'].upper()}")
    st.write(f"**Priority:** {agent_decision['priority']}")
    st.write(f"**Follow-up Window:** {agent_decision['follow_up_window']}")

    st.subheader("LLM Explanation & Recommendations")
    st.write(ai_msg.content)
