"""
ChurnGuard AI – Streamlit Chatbot
AI-powered retention assistant that predicts churn in real time
and recommends personalised retention strategies via GPT.
"""

import sys
from pathlib import Path
from typing import cast

# Ensure project root and app dir are on sys.path so imports work
_app_dir = str(Path(__file__).resolve().parent)
_root_dir = str(Path(__file__).resolve().parent.parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import streamlit as st
from openai.types.chat import ChatCompletionMessageParam
from churn_service import predict_churn, CUSTOMER_FIELDS, get_model_metadata
from llm_client import get_retention_advice, chat_general

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-low      { color: #27ae60; font-weight: bold; font-size: 1.3em; }
    .risk-medium   { color: #f39c12; font-weight: bold; font-size: 1.3em; }
    .risk-high     { color: #e74c3c; font-weight: bold; font-size: 1.3em; }
    .risk-veryhigh { color: #c0392b; font-weight: bold; font-size: 1.5em; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 15px;
        text-align: center; border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "churn_result" not in st.session_state:
    st.session_state.churn_result = None
if "customer_data" not in st.session_state:
    st.session_state.customer_data = {}

# ── Sidebar – Customer data form ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield--v1.png", width=60)
    st.title("ChurnGuard AI")
    st.caption("AI-Powered Customer Retention Assistant")
    st.divider()

    st.subheader("📋 Customer Profile")
    st.caption("Fill in the customer details to predict churn risk.")

    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, step=50.0)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

    if submitted:
        st.session_state.customer_data = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaymentMethod": payment,
            "PaperlessBilling": paperless,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        st.session_state.churn_result = predict_churn(st.session_state.customer_data)
        # Reset chat for new customer
        st.session_state.messages = []

    # Model info (collapsed)
    with st.expander("ℹ️ Model Info"):
        try:
            meta = get_model_metadata()
            st.markdown(f"""
            - **Model**: {meta['model_type']}
            - **Components**: {', '.join(meta['ensemble_components'])}
            - **Test ROC-AUC**: {meta['test_roc_auc']:.4f}
            - **Trained**: {meta['training_date']}
            """)
        except Exception:
            st.info("Model metadata not available.")

# ── Main area ───────────────────────────────────────────────────────────────
st.title("🛡️ ChurnGuard AI – Retention Assistant")

# Show prediction results if available
result = st.session_state.churn_result
if result:
    # Risk banner
    prob = result["churn_probability"]
    risk = result["risk_level"]
    risk_class = {
        "Low": "risk-low", "Medium": "risk-medium",
        "High": "risk-high", "Very High": "risk-veryhigh"
    }.get(risk, "risk-medium")

    col_a, col_b, col_c = st.columns([2, 2, 3])
    with col_a:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
    with col_b:
        st.markdown(f'Risk Level: <span class="{risk_class}">{risk}</span>',
                    unsafe_allow_html=True)
    with col_c:
        if result["risk_factors"]:
            st.markdown("**⚠️ Risk Factors:**")
            for f in result["risk_factors"]:
                st.markdown(f"- {f}")

    st.divider()

    # Auto-generate first AI message on new prediction
    if not st.session_state.messages:
        with st.spinner("🤖 Analysing customer and generating retention strategy..."):
            try:
                advice = get_retention_advice(result)
                st.session_state.messages.append(
                    {"role": "assistant", "content": advice}
                )
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": f"⚠️ Could not reach LLM: {e}\n\n"
                                f"**Churn probability: {prob*100:.1f}% ({risk})**\n\n"
                                f"Risk factors: {', '.join(result['risk_factors'])}"}
                )

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input for follow-up questions
    if prompt := st.chat_input("Ask about this customer or retention strategies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = cast(
                        list[ChatCompletionMessageParam],
                        [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ]
                    )
                    reply = get_retention_advice(
                        result, user_message=prompt, conversation_history=history
                    )
                except Exception as e:
                    reply = f"⚠️ LLM error: {e}"
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

else:
    # No prediction yet – show welcome state
    st.info("👈 **Fill in customer details** in the sidebar and click "
            "**Predict Churn Risk** to get started.")

    st.markdown("---")
    st.subheader("💬 General Churn & Retention Chat")
    st.caption("You can also ask general questions about churn and retention strategies.")

    # General chat (no customer loaded)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about churn, retention, telecom KPIs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = cast(
                        list[ChatCompletionMessageParam],
                        [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ]
                    )
                    reply = chat_general(prompt, conversation_history=history)
                except Exception as e:
                    reply = f"⚠️ LLM error: {e}"
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
