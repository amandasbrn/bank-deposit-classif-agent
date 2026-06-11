# Bank Deposit Decision Support

This app helps bank marketing teams decide how to follow up with customers for term deposit campaigns.

It uses a trained machine learning model to estimate subscription likelihood, then applies simple decision rules to suggest the next outreach action.

The Streamlit UI is currently under maintenance, but the project is already set up for Streamlit deployment.

## Features

- Predicts whether a customer is likely to subscribe to a term deposit
- Accepts campaign and customer inputs such as job, education, balance, contact history, and interaction duration
- Recommends a follow-up action such as `call`, `email`, or `do_not_contact`
- Assigns a priority level and follow-up window
- Generates short, readable recommendations for business users

## Project files

- `app.py`: Streamlit app
- `api/_shared.py`: model loading, prediction, and recommendation logic
- `*.pkl`: trained model artifacts
- `data_model.csv`: source data for valid input options and ranges

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets are required. The app runs from the code and model files in this repository.

## Notes

- `runtime.txt` pins Python 3.11.
- Keep `job_edu_encoder.pkl`, `selected_feature.pkl`, and `adaboost_trained.pkl` in the repository root.
