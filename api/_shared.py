from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "adaboost_trained.pkl"
FEATURE_PATH = BASE_DIR / "selected_feature.pkl"
JOB_EDU_ENCODER_PATH = BASE_DIR / "job_edu_encoder.pkl"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "adaboost_feature_imp.csv"
DATA_PATH = BASE_DIR / "data_model.csv"


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
def load_job_edu_encoder():
    return joblib.load(JOB_EDU_ENCODER_PATH)


@lru_cache(maxsize=1)
def load_feature_importance():
    frame = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return [
        {"name": row["feature"], "importance": float(row["importance"])}
        for _, row in frame.iterrows()
    ]


@lru_cache(maxsize=1)
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
    available_options = metadata()["educationByJob"].get(job, [])
    if education not in available_options:
        raise ValueError(f"Unsupported job and education combination: {key}")

    encoded = load_job_edu_encoder().transform(
        pd.DataFrame({"job_edu": [key]})
    )["job_edu"].iloc[0]

    frame = pd.DataFrame(
        {
            "duration": [float(payload["duration"])],
            "poutcome_success": [int(payload["poutcome_success"])],
            "job_edu": [float(encoded)],
            "was_contacted_before": [int(payload["was_contacted_before"])],
            "contact_cellular": [int(payload["contact_cellular"])],
            "balance": [float(payload["balance"])],
            "previous": [int(payload["previous"])],
        }
    )
    return frame[load_selected_features()], float(encoded)


def build_recommendations(ml_output):
    decision = ml_output["agent_decision"]
    input_features = ml_output["input_features"]
    probability = ml_output["probability"]

    why_lines = []
    if decision["action"] == "call":
        why_lines.append(
            f"The model estimates a {probability:.0%} subscription likelihood, so this customer is worth direct follow-up."
        )
    elif decision["action"] == "email":
        why_lines.append(
            f"The model estimates a {probability:.0%} subscription likelihood, which supports a lower-cost email follow-up."
        )
    else:
        why_lines.append(
            f"The model estimates a {probability:.0%} subscription likelihood, so active outreach should stay low priority."
        )

    if input_features["poutcome_success"] == 1:
        why_lines.append(
            "A previous successful outreach increases confidence and justifies faster follow-up."
        )
    elif input_features["duration"] > 300:
        why_lines.append(
            "The last interaction was relatively long, which signals stronger engagement than a short contact."
        )
    elif input_features["previous"] > 0:
        why_lines.append(
            "The customer has prior campaign history, which gives the team useful context for the next touchpoint."
        )

    next_steps = []
    if decision["action"] == "call":
        next_steps.append(
            f"Call the customer within {decision['follow_up_window']} while the signal is still fresh."
        )
        next_steps.append(
            "Lead with the product fit and reference the customer's recent engagement."
        )
        if input_features["poutcome_success"] == 1:
            next_steps.append(
                "Acknowledge the previous positive response and move quickly to a clear offer."
            )
        if input_features["was_contacted_before"] == 0:
            next_steps.append(
                "Keep the first outreach concise and confirm the best time for a longer follow-up."
            )
    elif decision["action"] == "email":
        next_steps.append(
            f"Send a targeted email within {decision['follow_up_window']} with one clear call to action."
        )
        next_steps.append(
            "Use a concise subject line and highlight the main deposit benefit early in the message."
        )
        next_steps.append(
            "Track opens or replies before deciding whether the lead should move to phone outreach."
        )
    else:
        next_steps.append(
            "Do not prioritize manual outreach for this customer in the current campaign."
        )
        next_steps.append(
            "Keep the customer in a lower-cost nurture segment until a stronger signal appears."
        )
        if input_features["previous"] > 0:
            next_steps.append(
                "Review prior campaign notes before reintroducing the lead to a future cycle."
            )

    return "WHY THIS ACTION:\n" + "\n".join(
        f"- {line}" for line in why_lines[:2]
    ) + "\n\nNEXT STEPS:\n" + "\n".join(f"- {line}" for line in next_steps[:4])


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
        "recommendations": build_recommendations(ml_output),
    }
