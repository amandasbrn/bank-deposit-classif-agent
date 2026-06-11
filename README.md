# Bank Deposit Decision Support

This repository is a single Streamlit app for predicting whether a banking customer is likely to subscribe to a term deposit product.

## Project structure

- `app.py`: Streamlit entrypoint
- `api/_shared.py`: shared model loading, feature preparation, prediction, and recommendation logic
- `*.pkl`: trained model artifacts required at runtime
- `data_model.csv`: source data used to build form metadata such as valid jobs and education options

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets or external AI configuration are required. The app generates recommendations locally, so deployment is just the code plus the model files already in this repo.

## Notes

- `runtime.txt` pins Python 3.11 for compatibility with the current dependency set.
- `job_edu_encoder.pkl`, `selected_feature.pkl`, and `adaboost_trained.pkl` must stay in the repository root for inference to work.
- The app uses the trained model plus deterministic recommendation rules, which makes hosted deployment simpler and more reliable.
