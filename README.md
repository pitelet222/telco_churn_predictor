# 🛡️ Telco Customer Churn Prediction & AI Retention Chatbot

An end-to-end data science project that predicts customer churn for a telecommunications company and uses **Generative AI** to recommend personalised retention strategies in real time.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Performance](#model-performance)
- [GenAI Integration – ChurnGuard AI](#genai-integration--churnguard-ai)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Project Overview

### Business Problem
Customer churn is one of the most critical challenges for telecom companies. Acquiring a new customer costs **5-7x more** than retaining an existing one. This project builds a machine learning pipeline to:

1. **Predict** which customers are likely to churn
2. **Identify** the key risk factors driving churn
3. **Recommend** personalised retention strategies via an AI-powered chatbot

### Dataset
- **Source**: Telco Customer Churn dataset (7,043 customer records)
- **Target**: `Churn` (binary – Yes/No)
- **Features**: 19 original features covering demographics, account info, and subscribed services

---

## Key Findings

### EDA Insights

| Finding | Detail |
|---|---|
| Overall churn rate | ~26.5% of customers churned |
| Fiber optic churn | **41.89%** – highest among internet service types |
| Electronic check | **45.29%** churn rate – highest payment method |
| Two-year contracts | Only **2.83%** churn – strongest loyalty indicator |
| No internet service | **7.40%** churn – lowest risk segment |
| Paperless billing | **33.57%** churn – counterintuitively high |

### Critical Risk Factors (from correlation analysis)
1. Month-to-month contracts (no long-term commitment)
2. Fiber optic internet (price/quality mismatch)
3. Electronic check payment (less engagement)
4. Short tenure (< 12 months)
5. High monthly charges (> $70)
6. No tech support or online security add-ons

---

## Project Structure

```
telco-churn-prediction/
├── app/                          # GenAI Chatbot Application
│   ├── app.py                    # Streamlit UI (sidebar form + chat)
│   ├── churn_service.py          # ML inference service (model loading + prediction)
│   └── llm_client.py             # OpenAI GPT integration (retention advice)
│
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned & feature-engineered dataset
│
├── models/                       # Trained model artifacts
│   ├── logistic_regression.pkl
│   ├── gradient_boosting.pkl
│   ├── catboost_model.pkl
│   ├── scaler.pkl                # StandardScaler (fitted on training data)
│   └── model_metadata.json       # Performance metrics & model info
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Initial data exploration
│   ├── 02_data_cleaning.ipynb        # Cleaning + feature engineering
│   ├── 03_eda.ipynb                  # Exploratory data analysis + visualizations
│   └── 04_modeling.ipynb             # Model training, evaluation & selection
│
├── src/                          # Source modules
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── reports/figures/              # Saved plots and visualizations
├── requirements.txt
├── .env.example                  # API key template
└── README.md
```

---

## Data Pipeline

### 1. Data Cleaning (`02_data_cleaning.ipynb`)
- Converted `TotalCharges` from string to numeric
- Handled 11 missing values (new customers with `tenure=0` → filled with 0)
- Removed duplicate rows
- Detected outliers via IQR method (kept as valid extreme behaviors)

### 2. Encoding Strategy
| Type | Columns | Method |
|---|---|---|
| Binary (Yes/No) | Partner, Dependents, PhoneService, PaperlessBilling | Yes=1, No=0 |
| Multi-category services | OnlineSecurity, TechSupport, StreamingTV, etc. | One-hot encoding (3 categories preserved) |
| Gender | Female/Male | Binary (0/1) |
| Categorical | InternetService, Contract, PaymentMethod | One-hot with `drop_first=True` |

### 3. Feature Engineering
| Feature | Formula | Purpose |
|---|---|---|
| `AvgMonthlyCharges` | TotalCharges / tenure | Average spending behavior |
| `TenureGroup` | Binned into 0-1yr, 1-2yr, 2-4yr, 4-6yr | Capture non-linear tenure effects |
| `TotalServices` | Count of active service subscriptions | Customer engagement indicator |

### 4. Final Dataset
- **35 numerical features** (all encoded + engineered)
- **70/30 train-test split** with stratification
- **StandardScaler** fitted on training data only

---

## Model Performance

### Models Evaluated

| Rank | Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 🥇 | **Ensemble (LR + GB + CatBoost)** | **0.8470** | 80.22% | 66.74% | 50.80% | 0.577 |
| 🥈 | CatBoost | 0.8452 | – | – | – | – |
| 🥉 | Logistic Regression | 0.8446 | – | – | – | – |
| 4 | Gradient Boosting | 0.8396 | – | – | – | – |
| 5 | XGBoost | 0.8358 | – | – | – | – |
| 6 | LightGBM | 0.8324 | – | – | – | – |
| 7 | Random Forest | 0.8278 | – | – | – | – |

### Selected Model: Soft Voting Ensemble
- **Strategy**: Average predicted probabilities from 3 best models
- **Components**: Logistic Regression + Gradient Boosting + CatBoost
- **Cross-validation**: 5-fold Stratified CV, ROC-AUC = 0.8476 ± 0.012
- **Stability**: Train ≈ Test performance (no overfitting detected)

### Why This Model?
- All 6 models performed within 0.6% ROC-AUC of each other → data is highly linearly separable
- Ensemble adds +0.24% improvement via model diversity
- Simple averaging is robust and fast for production inference

---

## GenAI Integration – ChurnGuard AI

### What It Is
An **AI-powered retention chatbot** built with Streamlit + OpenAI GPT-4o-mini that helps customer support agents predict churn risk and get actionable retention strategies in real time.

### Architecture
```
User fills customer form (sidebar)
        ↓
churn_service.py → encodes features → scales → 3 models predict → average probability
        ↓
Returns: churn_probability, risk_level, risk_factors, customer_summary
        ↓
llm_client.py → sends prediction context to GPT-4o-mini (retention specialist persona)
        ↓
GPT generates: risk explanation + 2-3 retention actions + agent script
        ↓
app.py renders everything in Streamlit chat UI (supports follow-up questions)
```

### Features
- **Real-time churn scoring**: Fill customer profile → instant prediction
- **Risk factor detection**: Highlights specific churn drivers (contract type, payment method, tenure, etc.)
- **AI retention advice**: GPT generates personalised strategies with estimated impact
- **Agent scripts**: Ready-to-use phrases for customer conversations
- **Multi-turn chat**: Follow-up questions like "What if we offer a 1-year contract?"
- **General chat mode**: Ask about churn KPIs and retention best practices without a customer loaded

### Components

| File | Role |
|---|---|
| `app/churn_service.py` | Loads trained models, encodes customer input (35 features), runs soft-voting prediction, detects risk factors |
| `app/llm_client.py` | OpenAI integration with "ChurnGuard AI" retention persona, supports multi-turn history |
| `app/app.py` | Streamlit UI – sidebar form, prediction display, chat interface |

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/pitelet222/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your OpenAI API key
copy .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-your-key-here

# 5. Run the chatbot
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Fill in customer details** in the sidebar (gender, tenure, contract, services, charges, etc.)
2. Click **"🔍 Predict Churn Risk"**
3. View the **churn probability**, **risk level** (Low/Medium/High/Very High), and **risk factors**
4. Read the **AI-generated retention strategy** with specific actions and an agent script
5. Ask **follow-up questions** in the chat (e.g., "What discount should we offer?")

---

## Technologies Used

| Category | Tools |
|---|---|
| **Data Analysis** | Pandas, NumPy, SciPy, Statsmodels |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| **Generative AI** | OpenAI GPT-4o-mini |
| **Web App** | Streamlit |
| **Environment** | python-dotenv |
| **Serialisation** | Joblib (model persistence) |

---

## Author

Developed as a final project for the Ironhack Data Science Bootcamp (2026).
