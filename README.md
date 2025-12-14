<h1 align="center">HalluciDetect</h1>

<p align="center">
  <img src="HalluciDetect.jpeg?v=2" alt="HalluciDetect Logo" width="800"/>
</p>

<p align="center">
  <strong>Automated LLM Evaluation & Hallucination Detector</strong>
</p>

<p align="center">
  <em>Version 1.3.0</em>
</p>

<p align="center">
  A comprehensive pipeline for automatically evaluating LLM outputs for hallucinations using Google Fact Check API, Wikipedia, custom knowledge bases, semantic similarity, and rule-based scoring. Features a professional dashboard with user authentication, JWT API access, prompt templates, PDF reports, and hallucination rate trends across models.
</p>

<p align="center">
  <a href="https://hallucidetect.onrender.com">🚀 Live Demo</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#api-documentation">API Docs</a>
</p>

---

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI[Web Dashboard]
        API_Client[API Client]
    end

    subgraph Server["Application Layer"]
        Flask[Flask Server]
        Evaluator[HallucinationEvaluator]
    end

    subgraph Evaluation["Evaluation Engine"]
        FC[Fact Checker]
        SS[Semantic Similarity]
        RB[Rule-Based Scorer]
    end

    subgraph External["External Services"]
        LLM[LLM Providers]
        FactAPI[Google Fact Check API]
        WikiAPI[Wikipedia API]
    end

    subgraph Storage["Data Layer"]
        DB[(PostgreSQL/SQLite)]
        KB[Knowledge Base]
    end

    UI --> Flask
    API_Client --> Flask
    Flask --> Evaluator
    Evaluator --> FC
    Evaluator --> SS
    Evaluator --> RB
    Evaluator --> LLM
    FC --> FactAPI
    Flask --> DB
    
    style Client fill:#1e1e2e,stroke:#6366f1,color:#f1f5f9
    style Server fill:#1e1e2e,stroke:#8b5cf6,color:#f1f5f9
    style Evaluation fill:#1e1e2e,stroke:#10b981,color:#f1f5f9
    style External fill:#1e1e2e,stroke:#f59e0b,color:#f1f5f9
    style Storage fill:#1e1e2e,stroke:#06b6d4,color:#f1f5f9
```

---

## Evaluation Pipeline Flow

```mermaid
flowchart LR
    A[Input Prompt] --> B[LLM Generation]
    B --> C[LLM Output]
    C --> D{Evaluation Engine}
    
    D --> E[Fact Checking]
    D --> F[Semantic Similarity]
    D --> G[Rule-Based Analysis]
    
    E --> H[Claim Extraction]
    H --> I[Claim Verification]
    I --> J[Fact Score]
    
    F --> K[TF-IDF Vectorization]
    K --> L[Cosine Similarity]
    L --> M[Semantic Score]
    
    G --> N[Pattern Detection]
    N --> O[Entity Consistency]
    O --> P[Rule Score]
    
    J --> Q{Score Aggregator}
    M --> Q
    P --> Q
    
    Q --> R[Overall Hallucination Score]
    R --> S{Threshold Check}
    S -->|Score >= 0.5| T[Hallucination Detected]
    S -->|Score < 0.5| U[Verified Output]
    
    T --> V[(Store Result)]
    U --> V
    
    style A fill:#6366f1,stroke:#6366f1,color:#fff
    style C fill:#8b5cf6,stroke:#8b5cf6,color:#fff
    style R fill:#f59e0b,stroke:#f59e0b,color:#fff
    style T fill:#ef4444,stroke:#ef4444,color:#fff
    style U fill:#10b981,stroke:#10b981,color:#fff
```

---

## Component Architecture

```mermaid
flowchart TB
    subgraph app["app.py - Flask Application"]
        routes[Route Handlers]
        api[REST API Endpoints]
    end

    subgraph evaluator["evaluator.py - Main Pipeline"]
        eval_single[evaluate]
        eval_batch[evaluate_batch]
        score_calc[_calculate_overall_score]
    end

    subgraph llm["llm_client.py - LLM Integration"]
        openrouter[OpenRouter Client]
        openai[OpenAI Client]
        anthropic[Anthropic Client]
        model_map[Model Mapping]
    end

    subgraph fact["fact_checker.py"]
        extract[_extract_claims]
        check[_check_single_claim]
        api_check[_check_via_api]
    end

    subgraph semantic["semantic_similarity.py"]
        tfidf[TF-IDF Vectorizer]
        similarity[compute_similarity]
        batch_sim[compute_similarity_batch]
    end

    subgraph rules["rule_based_scorer.py"]
        patterns[Suspicious Patterns]
        indicators[Positive Indicators]
        consistency[Entity Consistency]
    end

    subgraph database["database.py"]
        save[save_result]
        query[get_results]
        trends[get_trends]
    end

    subgraph models["models.py"]
        EvalResult[EvaluationResult]
        EvalBatch[EvaluationBatch]
    end

    routes --> eval_single
    routes --> eval_batch
    api --> query

    eval_single --> openrouter
    eval_single --> extract
    eval_single --> similarity
    eval_single --> patterns
    eval_single --> score_calc
    eval_single --> save

    eval_batch --> eval_single
    
    save --> EvalResult
    query --> EvalResult
```

---

## Scoring Algorithm

```mermaid
flowchart TB
    subgraph Input
        LLM_OUT[LLM Output]
        REF[Reference Text]
    end

    subgraph FactCheck["Fact Check Score (40%)"]
        FC1[Extract Claims]
        FC2[Verify Each Claim]
        FC3["fact_score = verified / total"]
        FC4["inverted = 1 - fact_score"]
    end

    subgraph Semantic["Semantic Score (30%)"]
        SS1[TF-IDF Vectorization]
        SS2[Cosine Similarity]
        SS3["semantic_score = similarity"]
        SS4["inverted = 1 - semantic_score"]
    end

    subgraph RuleBased["Rule-Based Score (30%)"]
        RB1[Detect Suspicious Patterns]
        RB2[Find Positive Indicators]
        RB3[Check Entity Consistency]
        RB4["rule_score = weighted_average"]
    end

    subgraph Aggregation
        AGG["overall = (fact × 0.4) + (semantic × 0.3) + (rule × 0.3)"]
        THRESHOLD{{"overall >= 0.5?"}}
        HALL[Hallucination: TRUE]
        VERIFIED[Hallucination: FALSE]
    end

    LLM_OUT --> FC1
    REF --> FC1
    FC1 --> FC2 --> FC3 --> FC4

    LLM_OUT --> SS1
    REF --> SS1
    SS1 --> SS2 --> SS3 --> SS4

    LLM_OUT --> RB1
    REF --> RB1
    RB1 --> RB2 --> RB3 --> RB4

    FC4 --> AGG
    SS4 --> AGG
    RB4 --> AGG

    AGG --> THRESHOLD
    THRESHOLD -->|Yes| HALL
    THRESHOLD -->|No| VERIFIED

    style HALL fill:#ef4444,color:#fff
    style VERIFIED fill:#10b981,color:#fff
```

---

## Database Schema

```mermaid
erDiagram
    USERS {
        string id PK
        string username UK
        string email UK
        string password_hash
        datetime created_at
    }

    EVALUATION_RESULTS {
        string id PK
        string user_id FK
        text prompt
        text llm_output
        string model_name
        string prompt_version
        datetime timestamp
        float semantic_similarity_score
        float fact_check_score
        float rule_based_score
        float overall_hallucination_score
        json fact_check_details
        json semantic_similarity_details
        json rule_based_details
        boolean is_hallucination
        float confidence
        json evaluation_metadata
    }

    EVALUATION_BATCHES {
        string batch_id PK
        text prompt_template
        string model_name
        string prompt_version
        datetime created_at
        integer total_evaluations
        integer hallucination_count
        float hallucination_rate
        json average_scores
    }

    PROMPT_TEMPLATES {
        string id PK
        string user_id FK
        string name
        text prompt_template
        text description
        string category
        text reference_template
        boolean is_public
        datetime created_at
    }

    USERS ||--o{ EVALUATION_RESULTS : owns
    USERS ||--o{ PROMPT_TEMPLATES : creates
    EVALUATION_BATCHES ||--o{ EVALUATION_RESULTS : contains
```

---

## API Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant F as Flask Server
    participant E as Evaluator
    participant L as LLM Client
    participant S as Scorers
    participant D as Database

    C->>F: POST /api/evaluate
    Note over C,F: {prompt, model_name, reference_text}
    
    F->>E: evaluate(prompt, model, reference)
    E->>L: generate(prompt, model)
    L-->>E: llm_output
    
    par Parallel Scoring
        E->>S: fact_checker.check_facts()
        E->>S: semantic_similarity.compute()
        E->>S: rule_scorer.score()
    end
    
    S-->>E: individual_scores
    E->>E: calculate_overall_score()
    E->>D: save_result()
    D-->>E: saved
    E-->>F: EvaluationResult
    F-->>C: JSON Response
```

---

## UI Navigation Structure

```mermaid
flowchart TB
    subgraph Public["Public Pages"]
        LAND[Landing Page]
        LOGIN[Login]
        REG[Register]
    end

    subgraph Sidebar["Sidebar Navigation"]
        direction TB
        DASH[Dashboard]
        PLAY[Playground]
        EVALS[Evaluations]
        ANAL[Analytics]
        TEMP[Templates]
        SETT[Settings]
        PROF[Profile]
    end

    subgraph Dashboard_Page["Dashboard"]
        D1[Metric Cards]
        D2[Trend Chart]
        D3[Distribution Chart]
        D4[Recent Results Table]
    end

    subgraph Playground_Page["Playground"]
        P1[Model Selection]
        P2[Template Selection]
        P3[Prompt Input]
        P4[Reference Input]
        P5[Evaluation Results]
        P6[Score Breakdown]
    end

    subgraph Evaluations_Page["Evaluations"]
        E1[Search & Filters]
        E2[Results Table]
        E3[CSV Import Modal]
        E4[Detail Modal]
        E5[CSV Export]
    end

    subgraph Templates_Page["Templates"]
        T1[Pre-built Templates]
        T2[Custom Templates]
        T3[Category Filter]
        T4[Use Template Modal]
    end

    subgraph Profile_Page["Profile"]
        PR1[User Details]
        PR2[Account Stats]
        PR3[Sign Out]
    end

    LAND --> LOGIN
    LAND --> REG
    LOGIN --> DASH
    REG --> DASH
    
    DASH --> Dashboard_Page
    PLAY --> Playground_Page
    EVALS --> Evaluations_Page
    TEMP --> Templates_Page
    PROF --> Profile_Page
```

---

## Tech Stack

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#fff', 'primaryBorderColor': '#818cf8', 'secondaryColor': '#8b5cf6', 'secondaryTextColor': '#fff', 'tertiaryColor': '#a855f7', 'tertiaryTextColor': '#fff', 'lineColor': '#94a3b8', 'textColor': '#fff'}}}%%
mindmap
    root((HalluciDetect))
        Backend
            Flask
            Flask-Login
            SQLAlchemy
            Gunicorn
            Python 3.9+
        Frontend
            HTML5/CSS3
            JavaScript
            Plotly.js
            CSS Animations
        AI/ML
            OpenRouter API
            TF-IDF Similarity
            Wikipedia API
            OpenAI SDK
            Anthropic SDK
        Database
            SQLite
            User Management
        Security
            Password Hashing
            Session Management
            Protected Routes
        Evaluation
            Fact Checking
            Semantic Similarity
            Rule-Based Scoring
```

---

## Features

### Core Evaluation
- **Multi-method Evaluation**: Combines fact-checking, semantic similarity, and rule-based scoring
- **LLM Integration**: Supports OpenAI, Anthropic, and 100+ models via OpenRouter
- **Google Fact Check API**: FREE fact verification using Google's Fact Check Tools
- **Wikipedia API Integration**: Enhanced entity verification using Wikipedia knowledge
- **Custom Knowledge Base**: Upload and manage your own verified facts for domain-specific evaluation
- **Batch Evaluation**: Evaluate multiple test cases at once
- **Trend Analysis**: Track hallucination rates across prompt versions and models

### User Experience
- **User Authentication**: Secure registration, login, and session management with Flask-Login
- **JWT API Authentication**: Token-based API access for programmatic integrations
- **User Profiles**: Personal dashboard with account statistics and evaluation history
- **Prompt Templates Library**: Pre-built and custom templates for common evaluation scenarios
- **CSV/JSON Import/Export**: Bulk import test cases and export evaluation results
- **PDF Reports**: Generate professional PDF reports of evaluation results
- **Professional Landing Page**: Animated, interactive introduction with live demo

### Advanced Evaluation (Phase 2)
- **NLP Metrics**: BLEU, ROUGE (1/2/L), METEOR, and Word Error Rate scores
- **A/B Prompt Testing**: Statistical comparison with Welch's t-test and Cohen's d effect size
- **RAG Evaluation**: Context relevance, answer faithfulness, and groundedness scoring
- **Webhook Integrations**: Custom endpoints and Slack notifications
- **Alert Rules**: Threshold-based alerts with configurable severity levels

### Technical
- **PostgreSQL Support**: Production-ready database with connection pooling
- **SQLite Fallback**: Zero-config local development database
- **Interactive Dashboard**: Professional dark-themed web interface with Plotly.js charts
- **Protected Routes**: All API endpoints secured with authentication
- **Database Backups**: Built-in backup utilities for data safety
- **Open Redirect Prevention**: Security hardened authentication flow

## Project Structure

```
.
├── app.py                    # Flask web application with routes & auth
├── config.py                 # Configuration settings (PostgreSQL, JWT, etc.)
├── database.py               # Database models, CRUD, user management, pooling
├── evaluator.py              # Main evaluation pipeline orchestrator
├── fact_checker.py           # Fact extraction, Google/Wikipedia/KB verification
├── google_factcheck.py       # Google Fact Check Tools API integration
├── knowledge_base.py         # Custom knowledge base for verified facts
├── jwt_auth.py               # JWT token authentication for API
├── pdf_export.py             # PDF report generation with ReportLab
├── backup_utils.py           # Database backup and restore utilities
├── nlp_metrics.py            # BLEU, ROUGE, METEOR, WER scoring
├── ab_testing.py             # A/B prompt testing with statistics
├── rag_evaluator.py          # RAG evaluation (faithfulness, relevance)
├── webhooks.py               # Webhook and Slack notifications
├── llm_client.py             # Multi-provider LLM client
├── models.py                 # Data models (EvaluationResult, Batch)
├── rule_based_scorer.py      # Pattern and entity analysis
├── semantic_similarity.py    # TF-IDF based similarity (lightweight)
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── static/
│   ├── css/style.css         # Dashboard styling
│   └── favicon.ico           # Site favicon
├── knowledge_base/           # Custom facts storage (JSON files)
├── backups/                  # Database backups directory
└── templates/
    ├── auth/
    │   ├── login.html        # User login page
    │   ├── register.html     # User registration page
    │   ├── forgot_password.html  # Password reset request
    │   └── reset_password.html   # Password reset form
    ├── base.html             # Shared layout with sidebar
    ├── landing.html          # Public landing page with animations
    ├── dashboard.html        # Overview with metrics
    ├── playground.html       # Single evaluation interface
    ├── evaluations.html      # Results history with CSV import
    ├── analytics.html        # Trend charts
    ├── templates.html        # Prompt templates library
    ├── profile.html          # User profile & security settings
    └── settings.html         # Configuration panel
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/rishi02102017/HalluciDetect.git
cd HalluciDetect
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# For production (lightweight, works on free hosting tiers)
pip install -r requirements.txt

# For local development (includes sentence-transformers for better accuracy)
pip install -r requirements-dev.txt
```

4. **Set up environment variables**

Create a `.env` file:
```env
# LLM API Keys
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=              # Optional
ANTHROPIC_API_KEY=           # Optional

# Database (PostgreSQL for production, SQLite for development)
DATABASE_URL=postgresql://user:pass@host:5432/hallucidetect  # Production
# DATABASE_URL=sqlite:///./evaluation_results.db             # Development

# Database Connection Pool (PostgreSQL only)
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE=3600

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=3600   # 1 hour
JWT_REFRESH_TOKEN_EXPIRES=604800  # 7 days

# Knowledge Base
KNOWLEDGE_BASE_PATH=./knowledge_base
FLASK_ENV=development
FLASK_DEBUG=True
USE_LOCAL_EMBEDDINGS=false   # Set to 'true' for sentence-transformers (requires more RAM)
```

Get your OpenRouter API key at: https://openrouter.ai/keys

> **Note**: For production, always use a strong, unique `SECRET_KEY` for session security.

## Usage

### Running the Dashboard

```bash
python app.py
```

Open http://localhost:5001 in your browser.

### Python API

```python
from evaluator import HallucinationEvaluator

evaluator = HallucinationEvaluator()

# Single evaluation
result = evaluator.evaluate(
    prompt="What is the capital of France?",
    model_name="gpt-4o-mini",
    prompt_version="v1",
    reference_text="The capital of France is Paris."
)

print(f"Hallucination Score: {result.overall_hallucination_score}")
print(f"Is Hallucination: {result.is_hallucination}")
print(f"Confidence: {result.confidence}")
```

### REST API Endpoints

All API endpoints (except public ones) require authentication via session or JWT token.

#### Authentication Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/token` | Get JWT access & refresh tokens |
| POST | `/api/auth/refresh` | Refresh JWT access token |
| GET | `/api/auth/verify` | Verify JWT token validity |

#### Evaluation Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluate` | Evaluate a single prompt (session auth) |
| POST | `/api/v1/evaluate` | Evaluate a single prompt (JWT auth) |
| POST | `/api/evaluate/batch` | Evaluate multiple test cases |
| POST | `/api/evaluate/csv` | Evaluate from uploaded CSV file |

#### Results & Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/results` | Get evaluation results for current user |
| GET | `/api/results/export` | Export results as CSV |
| GET | `/api/results/export/json` | Export results as JSON |
| GET | `/api/results/export/pdf` | Export results as PDF report |
| GET | `/api/batches` | Get batch evaluations |
| GET | `/api/trends` | Get hallucination trends |

#### Templates
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/templates` | Get prompt templates |
| POST | `/api/templates` | Create new template |
| GET | `/api/templates/<id>` | Get specific template |
| DELETE | `/api/templates/<id>` | Delete a template |
| GET | `/api/templates/export/json` | Export templates as JSON |
| POST | `/api/templates/import/json` | Import templates from JSON |

#### Knowledge Base
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/knowledge-base/search` | Search knowledge base |
| POST | `/api/knowledge-base/verify` | Verify a claim |
| POST | `/api/knowledge-base/facts` | Add a new fact |
| GET | `/api/knowledge-base/stats` | Get KB statistics |

#### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | Get available models (public) |
| GET | `/api/user` | Get current user info |
| POST | `/api/backup` | Create database backup |
| GET | `/api/backup/list` | List available backups |
| GET | `/health` | Health check (public) |

#### NLP Metrics (Phase 2)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/metrics/nlp` | Calculate all NLP metrics (BLEU, ROUGE, etc.) |
| POST | `/api/metrics/bleu` | Calculate BLEU score only |
| POST | `/api/metrics/rouge` | Calculate ROUGE scores only |

#### A/B Testing (Phase 2)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ab-tests` | List all A/B tests |
| POST | `/api/ab-tests` | Create new A/B test |
| GET | `/api/ab-tests/:id` | Get test details with analysis |
| POST | `/api/ab-tests/:id/results` | Add result to variant |

#### RAG Evaluation (Phase 2)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rag/evaluate` | Full RAG evaluation |
| POST | `/api/rag/context-relevance` | Check context relevance |
| POST | `/api/rag/faithfulness` | Check answer faithfulness |

#### Webhooks & Alerts (Phase 2)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/webhooks` | List all webhooks |
| POST | `/api/webhooks` | Create new webhook |
| DELETE | `/api/webhooks/:id` | Delete a webhook |
| POST | `/api/webhooks/test` | Send test webhook |
| GET | `/api/alerts` | List all alert rules |
| POST | `/api/alerts` | Create new alert rule |
| POST | `/api/slack/configure` | Configure Slack webhook |
| POST | `/api/slack/test` | Send test Slack message |

### JWT Authentication Example

```bash
# Get JWT token
curl -X POST https://hallucidetect.onrender.com/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword"}'

# Use token for API calls
curl -X POST https://hallucidetect.onrender.com/api/v1/evaluate \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "model_name": "gpt-4o-mini"}'
```

## Evaluation Methods

### 1. Fact-Checking (40% weight)

Multi-source fact verification using a cascade of methods:

1. **Reference Text Check** - Compare against provided reference (highest priority)
2. **Google Fact Check API** - FREE API from Google's Fact Check Tools
3. **Custom Knowledge Base** - Your own verified facts (domain-specific)
4. **Wikipedia API** - Entity verification using Wikipedia
5. **Heuristic Analysis** - Pattern-based claim analysis (fallback)

### 2. Semantic Similarity (30% weight)
- Uses lightweight TF-IDF by default (production-ready, no GPU required)
- Optional: sentence-transformers for higher accuracy (set `USE_LOCAL_EMBEDDINGS=true`)
- Computes cosine similarity
- Threshold: 0.7 (configurable)

### 3. Rule-Based Scoring (30% weight)
- Detects suspicious patterns (overconfident statements)
- Identifies positive indicators (citations, specifics)
- Checks entity consistency (numbers, dates)

### Overall Score Formula
```
hallucination_score = (fact_inverted × 0.4) + (semantic_inverted × 0.3) + (rule_score × 0.3)
```

## Dashboard Pages

| Page | Features |
|------|----------|
| **Landing** | Animated hero, live demo, feature showcase, FAQ |
| **Dashboard** | Metrics cards, trend charts, recent evaluations |
| **Playground** | Interactive evaluation with real-time results, template selection |
| **Evaluations** | Searchable history, filters, CSV import/export |
| **Analytics** | Model comparison, time-series trends |
| **Templates** | Pre-built prompts, custom template creation |
| **Profile** | User details, account statistics, sign out |
| **Settings** | API config, thresholds, database info |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HALLUCINATION_SCORE_THRESHOLD` | 0.5 | Score above this = hallucination |
| `SEMANTIC_SIMILARITY_THRESHOLD` | 0.7 | Minimum similarity for alignment |
| `FACT_CHECK_CONFIDENCE_THRESHOLD` | 0.8 | Confidence to verify a claim |
| `DEFAULT_LLM_MODEL` | gpt-4o-mini | Default model for evaluation |

## Deployment

### Environment Variables for Production
```env
OPENROUTER_API_KEY=your_key
DATABASE_URL=sqlite:///./evaluation_results.db
SECRET_KEY=generate-a-strong-secret-key
FLASK_ENV=production
FLASK_DEBUG=False
```

### Platforms
- **Render** (Recommended): Connect GitHub repo, auto-deploys on push
- **Railway**: `railway up`
- **Fly.io**: `fly launch`
- **Heroku**: `git push heroku main`

### Live Instance
The application is deployed at: **https://hallucidetect.onrender.com**

## Security

HalluciDetect implements several security best practices:

- **Password Hashing**: User passwords are hashed using PBKDF2-SHA256
- **Session Management**: Secure session handling with Flask-Login
- **Protected Routes**: All sensitive endpoints require authentication
- **Open Redirect Prevention**: Login redirects are validated to prevent phishing
- **User Data Isolation**: Users can only access their own evaluations and templates

## Contributing

This project demonstrates:
- LLM benchmarking and evaluation pipelines
- Prompt engineering with version tracking
- Python backend with Flask
- Professional dashboard development
- Data visualization with Plotly

---

## Author

**Jyotishman Das**

Built from scratch as a comprehensive LLM evaluation and hallucination detection platform.

---

## Changelog

### v1.1.0 (December 2025)
- **User Authentication**: Added registration, login, logout with Flask-Login
- **User Profiles**: Personal profile page with account statistics
- **Prompt Templates**: Pre-built and custom template library
- **CSV Operations**: Bulk import test cases, export evaluation results
- **Wikipedia API**: Enhanced fact verification using Wikipedia
- **Landing Page**: Professional animated landing page with live demo
- **Security**: Protected routes, open redirect prevention
- **UX Improvements**: Favicon, flash messages, loading spinners

### v1.0.0 (December 2025)
- Initial release with multi-method hallucination evaluation
- Support for OpenAI, Anthropic, and OpenRouter models
- Professional dark-themed dashboard
- Batch evaluation and trend analysis

---

## License

MIT License

Copyright (c) 2025 Jyotishman Das

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

**Attribution Requirement**: Any use, reproduction, or distribution of this 
software must include proper attribution to the original author (Jyotishman Das) 
with a link to the original repository.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
