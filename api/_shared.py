import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "adaboost_trained.pkl"
FEATURE_PATH = BASE_DIR / "selected_feature.pkl"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "adaboost_feature_imp.csv"
DATA_PATH = BASE_DIR / "data_model.csv"


SYSTEM_INSTRUCTION = """
You are an AI assistant supporting a bank's marketing operations team.

The outreach action has ALREADY been selected by a decision agent.
You must NOT change or suggest a different action.

Your role is to:
- Briefly explain why this action was chosen (1-2 sentences)
- Provide concise, action-oriented execution guidance

Style rules:
- Write for busy professionals
- Use short, direct sentences
- Focus on what to do and when
- Avoid unnecessary background or storytelling
- No emojis, no marketing fluff
""".strip()


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_selected_features():
    return joblib.load(FEATURE_PATH)


@lru_cache(maxsize=1)
def load_feature_importance():
    frame = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return [
        {"name": row["feature"], "importance": float(row["importance"])}
        for _, row in frame.iterrows()
    ]


@lru_cache(maxsize=1)
def job_education_lookup():
    data = load_data().copy()
    data["job_edu_key"] = data["job"].astype(str) + "_" + data["education"].astype(str)
    grouped = data.groupby("job_edu_key", as_index=False)["deposit"].mean()
    return dict(zip(grouped["job_edu_key"], grouped["deposit"]))


def metadata():
    data = load_data()
    jobs = sorted(data["job"].dropna().unique().tolist())
    education_by_job = {
        job: sorted(
            data.loc[data["job"] == job, "education"].dropna().unique().tolist()
        )
        for job in jobs
    }
    return {
        "jobs": jobs,
        "educationByJob": education_by_job,
        "ranges": {
            "duration": {
                "min": float(data["duration"].min()),
                "max": float(data["duration"].max()),
            },
            "balance": {
                "min": float(data["balance"].min()),
                "max": float(data["balance"].max()),
            },
            "previous": {
                "min": int(data["previous"].min()),
                "max": int(data["previous"].max()),
            },
        },
    }


def marketing_decision_agent(
    probability: float,
    poutcome_success: int,
    duration: float,
    was_contacted_before: int,
):
    decision = {
        "action": "do_not_contact",
        "priority": "low",
        "follow_up_window": None,
        "reason": "low_probability",
    }

    if probability >= 0.6:
        decision["action"] = "call"
        decision["priority"] = "high"
        decision["follow_up_window"] = "24-48 hours"
        decision["reason"] = "high_subscription_probability"
    elif 0.4 <= probability < 0.6:
        decision["action"] = "email"
        decision["priority"] = "medium"
        decision["follow_up_window"] = "2-3 days"
        decision["reason"] = "medium_subscription_probability"

    if poutcome_success == 1:
        decision["priority"] = "high"
        decision["reason"] += "_previous_success"

    if duration > 300:
        decision["reason"] += "_high_engagement"

    if was_contacted_before == 0 and decision["action"] == "call":
        decision["reason"] += "_first_outreach"

    return decision


def build_model_input(payload):
    job = str(payload["job"]).strip()
    education = str(payload["education"]).strip()
    key = f"{job}_{education}"
    lookup = job_education_lookup()
    if key not in lookup:
        raise ValueError(f"Unsupported job and education combination: {key}")

    frame = pd.DataFrame(
        {
            "duration": [float(payload["duration"])],
            "poutcome_success": [int(payload["poutcome_success"])],
            "job_edu": [float(lookup[key])],
            "was_contacted_before": [int(payload["was_contacted_before"])],
            "contact_cellular": [int(payload["contact_cellular"])],
            "balance": [float(payload["balance"])],
            "previous": [int(payload["previous"])],
        }
    )
    return frame[load_selected_features()], float(lookup[key])


def generate_explanation(ml_output):
    api_key = os.getenv("api_key")
    if not api_key:
        return (
            "WHY THIS ACTION:\n"
            "- Explanation unavailable because the Gemini API key is not configured.\n\n"
            "NEXT STEPS:\n"
            "- Add the `api_key` environment variable in Vercel before requesting AI guidance."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"""
Here is the system output in JSON:
{json.dumps(ml_output, ensure_ascii=False)}

Respond using EXACTLY this format:

WHY THIS ACTION (max 2 bullet points):
- ...
- ...

NEXT STEPS (max 4 bullet points):
- Start each bullet with a strong action verb
- Include timing if applicable
- Each bullet must be one sentence only
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"system_instruction": SYSTEM_INSTRUCTION, "temperature": 0.3},
    )
    return response.text


def predict(payload):
    model_input, job_edu_value = build_model_input(payload)
    model = load_model()
    prediction = int(model.predict(model_input)[0])
    probability = float(model.predict_proba(model_input)[0][1])
    agent_decision = marketing_decision_agent(
        probability=probability,
        poutcome_success=int(payload["poutcome_success"]),
        duration=float(payload["duration"]),
        was_contacted_before=int(payload["was_contacted_before"]),
    )
    ml_output = {
        "prediction": prediction,
        "prediction_label": "subscribe" if prediction == 1 else "not_subscribe",
        "probability": probability,
        "agent_decision": agent_decision,
        "top_features": load_feature_importance(),
        "input_features": {
            "duration": float(payload["duration"]),
            "poutcome_success": int(payload["poutcome_success"]),
            "job_edu": job_edu_value,
            "was_contacted_before": int(payload["was_contacted_before"]),
            "contact_cellular": int(payload["contact_cellular"]),
            "balance": float(payload["balance"]),
            "previous": int(payload["previous"]),
            "job": str(payload["job"]).strip(),
            "education": str(payload["education"]).strip(),
        },
    }
    return {
        "prediction": prediction,
        "predictionLabel": "Likely to subscribe" if prediction == 1 else "Unlikely to subscribe",
        "probability": probability,
        "agentDecision": agent_decision,
        "recommendations": generate_explanation(ml_output),
    }
