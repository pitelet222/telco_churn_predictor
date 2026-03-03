# 🛡️ Telco Customer Churn Prediction & AI Retention Chatbot

An end-to-end data science project that predicts customer churn for a telecommunications company and uses **Generative AI** to recommend personalised retention strategies in real time.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-009688)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![Tests](https://img.shields.io/badge/Tests-172%20passed-brightgreen)
![CI](https://img.shields.io/github/actions/workflow/status/pitelet222/telco_churn_predictor/ci.yml?label=CI)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Performance](#model-performance)
- [GenAI Integration – ChurnGuard AI](#genai-integration--churnguard-ai)
- [REST API](#rest-api)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Docker](#docker)
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
│   └── llm_client.py             # OpenAI GPT integration (sync + async)
│
├── api/                          # REST API (FastAPI)
│   ├── main.py                   # App assembly (CORS, auth middleware, routers)
│   ├── schemas.py                # Pydantic request/response models
│   └── routers/
│       ├── health.py             # GET /health, GET /model/metadata
│       ├── predict.py            # POST /predict
│       └── advice.py             # POST /advice (async)
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned & feature-engineered dataset
│   └── tableau/                  # Tableau-ready export
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
├── scripts/
│   └── export_tableau_data.py    # Export cleaned data for Tableau dashboards
│
├── tests/                        # Test suite (172 tests)
│   ├── conftest.py               # Shared fixtures (customer profiles)
│   ├── test_api.py               # API endpoint tests (40 tests)
│   ├── test_churn_service.py     # Core prediction tests
│   ├── test_config.py            # Configuration tests
│   ├── test_evaluate.py          # Evaluation utils tests
│   ├── test_log_config.py        # Logging tests
│   └── test_train.py             # Training pipeline tests
│
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── pyproject.toml                # Package metadata & editable install config
├── config.py                     # Centralized settings (Pydantic BaseSettings)
├── log_config.py                 # Rotating file + console logging
├── Dockerfile                    # Multi-stage production image (API)
├── docker-compose.yml            # One-command API deployment
├── requirements.txt              # Full dependencies (ML + app + API)
├── requirements-api.txt          # API-only dependencies (lighter image)
├── reports/figures/              # Saved plots and visualizations
├── .env.example                  # Environment variable template
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
| `app/llm_client.py` | OpenAI integration (sync + async clients) with "ChurnGuard AI" retention persona, supports multi-turn history |
| `app/app.py` | Streamlit UI – sidebar form, prediction display, chat interface |

---

## REST API

### What It Is
A **FastAPI REST service** that exposes the churn prediction model over HTTP, allowing any application (CRM systems, mobile apps, automated scripts) to get predictions programmatically — without the Streamlit UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok", "version": "1.0.0"}` |
| `GET` | `/model/metadata` | Training metrics, model components, and training date |
| `POST` | `/predict` | Accepts a customer profile, returns churn probability + risk level + SHAP factors |
| `POST` | `/advice` | Runs prediction + calls GPT for personalised retention strategies |

### Architecture
```
Client (any app) → POST /predict → schemas.py validates input
     → churn_service.py encodes + scales + 3-model ensemble → PredictionResponse

Client → POST /advice → prediction step above
     → llm_client.py sends context to GPT → AdviceResponse (prediction + advice)
```

### Key Design Decisions
- **Model preloading**: Artifacts load at startup (lifespan handler), so the first request isn't slow
- **No logic duplication**: Routers call the same `predict_churn()` and `get_retention_advice()` used by Streamlit
- **Async /advice**: The advice endpoint uses `AsyncOpenAI` for non-blocking LLM calls under load
- **API key auth**: Optional `X-API-Key` header middleware — activated when `API_KEY` is set in environment
- **Strict validation**: `Literal` types reject invalid inputs before they reach the model (e.g., `"Contract": "Weekly"` → 422)
- **Structured errors**: 422 for validation, 500 for model failures, 502 for LLM failures
- **CORS with production warning**: `API_CORS_ORIGINS=["*"]` by default; logs a warning if left open in production
- **Interactive docs**: Auto-generated Swagger UI at `/docs`

### Running the API
```bash
uvicorn api.main:app --reload          # Development (auto-reload)
uvicorn api.main:app --workers 4       # Production (multiple workers)
```

---

## Testing

### Test Suite
The project includes **172 tests** covering the core logic, configuration, logging, training pipeline, and API endpoints.

| Test File | Tests | Coverage |
|---|---|---|
| `test_api.py` | 40 | All 4 API endpoints, input validation (422s), error handling, edge cases |
| `test_churn_service.py` | ~40 | Feature encoding, prediction pipeline, SHAP, fallback rules, summaries |
| `test_config.py` | ~18 | Default values, types, path validation, env overrides |
| `test_evaluate.py` | ~12 | Metrics computation, confusion matrix, ROC, model comparison |
| `test_log_config.py` | ~10 | Handler setup, file creation, log writing, suppression |
| `test_train.py` | ~10 | Data loading, splitting, scaling, stratification, leakage detection |

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run only API tests
python -m pytest tests/test_api.py -v

# Run with short traceback
python -m pytest tests/ -v --tb=short
```

---

## CI/CD

The project uses **GitHub Actions** to run the full test suite automatically on every push and pull request to `master`.

### Pipeline (`.github/workflows/ci.yml`)
```
Push to master → GitHub spins up Ubuntu VM → Installs Python 3.13 + dependencies → Runs pytest (172 tests) → Reports pass/fail
```

Results are visible as ✅/❌ on commits and pull requests in the GitHub repo.

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

# 3. Install dependencies + editable package
pip install -r requirements.txt
pip install -e .              # required — makes config, app, api, src importable

# 4. Set up your OpenAI API key
copy .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-your-key-here

# 5. Run the chatbot
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

### Streamlit Chatbot
1. **Fill in customer details** in the sidebar (gender, tenure, contract, services, charges, etc.)
2. Click **"🔍 Predict Churn Risk"**
3. View the **churn probability**, **risk level** (Low/Medium/High/Very High), and **risk factors**
4. Read the **AI-generated retention strategy** with specific actions and an agent script
5. Ask **follow-up questions** in the chat (e.g., "What discount should we offer?")

### REST API
1. Start the server: `uvicorn api.main:app --reload`
2. Open **Swagger docs**: `http://127.0.0.1:8000/docs`
3. Use **POST /predict** with a customer JSON payload to get a churn prediction
4. Use **POST /advice** to also get AI-generated retention recommendations

**Example (Python):**
```python
import requests

customer = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
    "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes", "MonthlyCharges": 85.50,
    "TotalCharges": 256.50,
}

response = requests.post("http://127.0.0.1:8000/predict", json=customer)
print(response.json())
# → {"churn_probability": 0.7576, "risk_level": "Very High", ...}
```

---

## Configuration

All settings live in `config.py` (Pydantic `BaseSettings`). They can be overridden via environment variables or a `.env` file — environment variables always take priority.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Required for chatbot & /advice endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ENVIRONMENT` | `development` | `development` \| `staging` \| `production` |
| `CHURN_THRESHOLD` | `0.5` | Classification cutoff (lower → higher recall) |
| `API_KEY` | `""` | If set, all API requests require `X-API-Key` header |
| `API_CORS_ORIGINS` | `["*"]` | Restrict in production |
| `SHAP_TOP_N` | `5` | Number of SHAP factors shown |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See `.env.example` for the full list with defaults.

---

## Docker

The API can be deployed as a Docker container using the included multi-stage `Dockerfile` and `docker-compose.yml`.

```bash
# Build and start the API
docker compose up --build -d

# Check it's running
curl http://localhost:8000/health
# → {"status": "ok", "version": "1.0.0"}

# Stop
docker compose down
```

The image uses `requirements-api.txt` (lighter than the full `requirements.txt`) and runs Uvicorn on port 8000. Environment variables are loaded from `.env` via `env_file` in `docker-compose.yml`.

---

## Technologies Used

| Category | Tools |
|---|---|
| **Data Analysis** | Pandas, NumPy, SciPy, Statsmodels |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-Learn, XGBoost, LightGBM, CatBoost, SHAP |
| **Generative AI** | OpenAI GPT-4o-mini |
| **Web App** | Streamlit |
| **REST API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest (172 tests) |
| **CI/CD** | GitHub Actions |
| **Configuration** | Pydantic Settings, python-dotenv |
| **Serialisation** | Joblib (model persistence) |

---

## Author

Developed as a final project for the Ironhack Data Science Bootcamp (2026).
