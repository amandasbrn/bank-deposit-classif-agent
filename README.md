# Agentic AI Marketing Decision System for Bank Deposits

An end-to-end agentic AI system designed to support bank marketing teams in making customer outreach decisions for term deposit products.  
The system combines a machine learning prediction model, a rule-based decision agent, and a large language model (LLM) to move from prediction to actionable marketing guidance.

---

## 🔍 Problem Overview
Banks often rely on predictive models to estimate whether a customer is likely to subscribe to a term deposit.  
However, predictions alone are not sufficient for operational decision-making.

This project addresses the gap by:
- Predicting subscription likelihood
- Selecting an appropriate outreach action using agentic logic
- Explaining decisions and providing concise, execution-ready recommendations

---

## 🧠 System Architecture
1. **Prediction Model**  
   A supervised classification model estimates the probability that a customer will subscribe to a term deposit.

2. **Decision Agent (Rule-Based)**  
   A deterministic agent applies business rules to select the optimal outreach action (e.g., call, email, or no contact) based on model output and customer context.

3. **LLM Explainer**  
   A large language model explains the agent’s decision in non-technical language and generates concise, action-oriented guidance for bank staff.

---

## ⚙️ Key Features
- End-to-end ML pipeline from feature engineering to inference
- Explicit separation between prediction, decision, and explanation
- Rule-based agentic decision policy for transparency and auditability
- LLM output constrained for professional, concise, and urgent communication
- Interactive Streamlit interface for real-time predictions and explanations

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Machine Learning:** Scikit-learn  
- **Agent Logic:** Rule-based decision policy  
- **LLM:** Gemini (via LangChain)  
- **Frontend:** Streamlit  

---

## 🎯 Focus Areas
- Supervised Learning (Classification)
- Agentic AI (Decision Policies)
- Explainable AI
- Applied AI for Business Decision Support

---

## 🚀 Future Improvements
- Add feedback loop to simulate customer responses
- Learn decision policies from historical outcomes
- Extend agent memory across multiple customer interactions
