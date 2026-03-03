# Bank Deposit Decision Support

This project is now structured for Vercel:

- Static frontend: [`index.html`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/index.html)
- Python API: [`api/index.py`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/api/index.py)
- Shared inference logic: [`api/_shared.py`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/api/_shared.py)

## Local development

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
flask --app api.index run --debug
```

Then open [`index.html`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/index.html) through a simple static server, for example:

```bash
python -m http.server 3000
```

If you use a separate static server, point the frontend at the Flask API origin or proxy `/api/*` to the Flask server.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Import the repository into Vercel.
3. Set the project root to this repository.
4. Add environment variable `api_key` if you want Gemini-generated recommendations.
5. Deploy.

Vercel will:

- Serve the static frontend from the repository root
- Route `/api/*` to the Flask serverless function via [`vercel.json`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/vercel.json)
- Install Python dependencies from [`requirements.txt`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/requirements.txt)

## API contract

`GET /api/metadata`

Returns:

- Available job options
- Education options by job
- Observed numeric ranges for the form

`POST /api/predict`

Example body:

```json
{
  "job": "technician",
  "education": "secondary",
  "duration": 117,
  "poutcome_success": 0,
  "was_contacted_before": 0,
  "contact_cellular": 1,
  "balance": 7,
  "previous": 0
}
```

## Notes

- The original Streamlit app remains in [`app.py`](/Users/diraamanda/Documents/GitHub/bank-fraud-classif-llm/app.py), but Vercel uses the new static frontend and API.
- `job_edu_encoder.pkl` is no longer required at runtime. The API derives the `job_edu` value from the dataset to avoid the extra `category_encoders` dependency.
- If `api_key` is missing, predictions still work and the API returns a fallback recommendation message instead of Gemini output.
