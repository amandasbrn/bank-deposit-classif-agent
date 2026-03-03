from flask import Flask, jsonify, request

from api._shared import metadata, predict


app = Flask(__name__)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    return jsonify(metadata())


@app.route("/api/predict", methods=["POST"])
def post_predict():
    payload = request.get_json(silent=True) or {}
    required = [
        "job",
        "education",
        "duration",
        "poutcome_success",
        "was_contacted_before",
        "contact_cellular",
        "balance",
        "previous",
    ]
    missing = [field for field in required if field not in payload or payload[field] in ("", None)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        return jsonify(predict(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500
