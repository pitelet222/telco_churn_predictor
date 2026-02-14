"""
LLM Client – OpenAI Integration
Sends customer churn context to GPT and returns a retention-focused response.
"""

import sys
from pathlib import Path
from typing import Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Ensure project root is on sys.path so we can import config
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ── System prompt (retention specialist persona) ────────────────────────────
SYSTEM_PROMPT = """You are **ChurnGuard AI**, an expert customer-retention assistant 
for a telecommunications company. You have access to a real-time machine-learning 
churn prediction model.

Your responsibilities:
1. **Analyse** the churn prediction results and risk factors provided.
2. **Explain** the risk to a support agent in clear, non-technical language.
3. **Recommend** 2-3 concrete, personalised retention actions based on the 
   customer's profile (contract upgrade, discount, service bundle, loyalty reward, etc.).
4. **Estimate** the potential impact of each action (e.g., "switching to a 1-year 
   contract reduces churn probability by ~30%").
5. Keep responses concise (≤ 250 words), structured with bullet points, and 
   action-oriented.

Formatting rules:
- Use markdown for headings & bullets.
- Start with a one-line churn risk summary.
- End with a recommended script the agent could say to the customer.

If the user asks a general question (not about a specific customer), answer helpfully 
about churn, retention strategies, or telecom industry best practices."""


def get_retention_advice(
    churn_result: dict[str, Any],
    user_message: str = "",
    conversation_history: list[ChatCompletionMessageParam] | None = None,
) -> str:
    """
    Call OpenAI to generate retention advice given model output.

    Parameters
    ----------
    churn_result : dict
        Output from churn_service.predict_churn() containing:
        - churn_probability, risk_level, risk_factors, customer_summary
    user_message : str
        Optional additional question from the agent/user.
    conversation_history : list
        Previous messages for multi-turn context.

    Returns
    -------
    str  – Markdown-formatted retention advice from the LLM.
    """
    # Build the context message with prediction data
    prediction_context = (
        f"## Churn Prediction Results\n"
        f"- **Churn Probability**: {churn_result['churn_probability']*100:.1f}%\n"
        f"- **Risk Level**: {churn_result['risk_level']}\n"
        f"- **{churn_result['customer_summary']}**\n\n"
        f"### Risk Factors Detected:\n"
    )
    for factor in churn_result.get("risk_factors", []):
        prediction_context += f"- {factor}\n"

    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include conversation history (if multi-turn)
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current turn
    full_user_msg = prediction_context
    if user_message:
        full_user_msg += f"\n\n**Agent question**: {user_message}"
    else:
        full_user_msg += (
            "\n\nPlease analyse this customer and provide retention recommendations."
        )

    messages.append({"role": "user", "content": full_user_msg})

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    return response.choices[0].message.content or ""


def chat_general(user_message: str, conversation_history: list[ChatCompletionMessageParam] | None = None) -> str:
    """
    General-purpose chat (no prediction context).
    Useful when the user asks about churn strategies, KPIs, etc.
    """
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    return response.choices[0].message.content or ""
