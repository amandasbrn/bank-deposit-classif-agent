# Bank Deposit Decision Support

This project predicts whether a banking customer is likely to subscribe to a term deposit product.

The Streamlit UI is currently under maintenance, but the repository is already set up for Streamlit deployment.

## Files

- `app.py`: Streamlit entrypoint
- `api/_shared.py`: prediction and recommendation logic
- `*.pkl`: trained model files
- `data_model.csv`: data used for form metadata

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
