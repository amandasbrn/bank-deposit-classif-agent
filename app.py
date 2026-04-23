import streamlit as st
from api._shared import load_data, metadata, predict

st.header('🏦 AI-Powered Marketing Decision Support for Bank Deposits')
st.markdown(
    "*An end-to-end AI system that predicts customer subscription likelihood, "
    "selects outreach actions using rule-based agent logic, and explains decisions "
    "using a large language model. Built to reflect real-world banking workflows.*"
)
st.divider()

data = load_data()
meta = metadata()

# --- Sidebar inputs ---
input_poutcome_success = st.sidebar.radio(
    'Customer conversion in last outreach?',
    ["Successfully Converted", "Not Successful"]
)
poutcome_success = 1 if input_poutcome_success == "Successfully Converted" else 0

duration = st.sidebar.number_input(
    'How long did the last interaction last? (seconds)', value=None
)
st.sidebar.caption(f"min: {meta['ranges']['duration']['min']:.0f}, max: {meta['ranges']['duration']['max']:.0f}")

job = st.sidebar.selectbox('Occupation:', options=meta['jobs'])
edu_options = meta['educationByJob'].get(job, [])
edu = st.sidebar.selectbox('Latest education:', options=edu_options)

input_was_contacted_before = st.sidebar.radio(
    'Was the customer contacted before?', ["Yes", "No"]
)
was_contacted_before = 1 if input_was_contacted_before == "Yes" else 0

input_contact_cellular = st.sidebar.radio(
    'Was the customer contacted on cellular?', ["Yes", "No"]
)
contact_cellular = 1 if input_contact_cellular == "Yes" else 0

balance = st.sidebar.number_input("Customer's balance", value=None)
st.sidebar.caption(f"min: {meta['ranges']['balance']['min']:.0f}, max: {meta['ranges']['balance']['max']:.0f}")

previous = st.sidebar.number_input(
    'Times contacted in previous campaign?', value=None, step=1
)
st.sidebar.caption(f"min: {meta['ranges']['previous']['min']}, max: {meta['ranges']['previous']['max']}")

pred_button = st.sidebar.button("Get prediction", type="primary")

# --- Prediction ---
if pred_button:
    missing = [
        name for name, val in [
            ("duration", duration),
            ("balance", balance),
            ("previous", previous),
        ]
        if val is None
    ]
    if missing:
        st.error(f"Please fill in: {', '.join(missing)}")
    else:
        payload = {
            "job": job,
            "education": edu,
            "duration": duration,
            "poutcome_success": poutcome_success,
            "was_contacted_before": was_contacted_before,
            "contact_cellular": contact_cellular,
            "balance": balance,
            "previous": previous,
        }
        try:
            result = predict(payload)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if result["prediction"] == 1:
            st.success(
                f"{result['predictionLabel']} — Probability: {result['probability']:.2f}"
            )
        else:
            st.warning(
                f"{result['predictionLabel']} — Probability: {result['probability']:.2f}"
            )

        decision = result["agentDecision"]
        st.subheader("Agent Decision")
        st.write(f"**Action:** {decision['action'].upper()}")
        st.write(f"**Priority:** {decision['priority']}")
        st.write(f"**Follow-up Window:** {decision.get('follow_up_window') or 'N/A'}")

        st.subheader("Recommendations")
        st.write(result["recommendations"])
