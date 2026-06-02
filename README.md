# TrialBridge 🧬

> AI-powered clinical trial matching platform — connecting patients to eligible trials through intelligent EHR parsing, semantic search, and explainable matching.

[![CI](https://github.com/yourusername/trialbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/trialbridge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## The Problem

Over 80% of clinical trials fail to meet enrollment targets. Simultaneously, millions of patients who qualify for trials that could save their lives never hear about them. The matching process today is manual, slow, and error-prone — done by research coordinators reading eligibility criteria line by line.

**TrialBridge solves this.**

---

## What It Does

1. **Ingests** unstructured patient EHR notes (clinical text)
2. **Extracts** structured medical profile using NLP (conditions, medications, labs, demographics)
3. **Matches** against 400,000+ ClinicalTrials.gov trials using BioBERT semantic search + eligibility rules engine
4. **Explains** each match in plain English — why a patient qualifies, what flags exist, what to verify

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        TrialBridge                          │
│                                                             │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────┐  │
│  │ EHR Ingestion│──▶│  NLP Pipeline   │──▶│Match Engine │  │
│  │    Layer     │   │scispaCy+BioBERT │   │FAISS+Rules  │  │
│  └──────────────┘   └─────────────────┘   └──────┬──────┘  │
│                                                   │         │
│  ┌────────────────────────────────────────────────▼──────┐  │
│  │              Explanation Engine (Claude API)          │  │
│  └────────────────────────────────────────────────┬──────┘  │
│                                                   │         │
│  ┌─────────────────┐   ┌───────────────────────────▼─────┐  │
│  │  PostgreSQL DB  │   │     FastAPI Backend              │  │
│  │  + pgvector     │   │  JWT Auth + Rate Limiting        │  │
│  │  + Redis Cache  │   │  Audit Logging + OpenAPI         │  │
│  └─────────────────┘   └───────────────────────────┬─────┘  │
│                                                     │        │
│                                          ┌──────────▼─────┐  │
│                                          │Streamlit Front │  │
│                                          └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Database | PostgreSQL + pgvector |
| Cache | Redis |
| Task Queue | Celery |
| Medical NER | scispaCy (en_core_sci_lg) |
| Embeddings | BioBERT (HuggingFace) |
| Vector Search | pgvector + FAISS |
| Explanation | Claude API (Anthropic) |
| Frontend | Streamlit |
| Infrastructure | Docker + Docker Compose |
| Cloud | AWS (EC2, RDS, S3, ElastiCache) |
| CI/CD | GitHub Actions |
| Monitoring | Sentry + Prometheus + Grafana |
| Reverse Proxy | Nginx |

---

## Quick Start (Local Development)

### Prerequisites
- Docker Desktop installed
- Git

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/trialbridge.git
cd trialbridge
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start all services
```bash
docker compose up --build
```

### 4. Access the app
- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health
- **Grafana:** http://localhost:3000

---

## Project Structure

```
trialbridge/
├── backend/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/      # Route handlers
│   ├── core/
│   │   ├── config.py           # App configuration
│   │   ├── security.py         # JWT auth
│   │   ├── database.py         # DB connection
│   │   └── logging.py          # Structured logging
│   ├── models/
│   │   ├── trial.py            # Trial ORM model
│   │   ├── patient.py          # Patient profile model
│   │   ├── match.py            # Match result model
│   │   └── audit.py            # Audit log model
│   ├── nlp/                    # NLP pipeline (Week 2)
│   ├── matching/               # Match engine (Week 3)
│   └── tests/                  # Pytest test suite
├── frontend/                   # Streamlit app (Week 4)
├── infra/
│   ├── nginx/                  # Nginx config
│   └── docker/                 # Service Dockerfiles
├── scripts/
│   └── ingest_trials.py        # ClinicalTrials.gov ingestion
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/token` | Get JWT token |
| POST | `/api/v1/parse-patient` | Parse EHR note → structured profile |
| POST | `/api/v1/match` | Match patient to trials |
| GET | `/api/v1/trials/{id}` | Get trial details |
| GET | `/api/v1/matches/{id}` | Get match result with explanation |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

---

## Roadmap

### V1 (Current — 30 days)
- [x] Production infrastructure
- [ ] NLP patient extraction pipeline
- [ ] Semantic matching engine
- [ ] Eligibility rules parser
- [ ] Explanation engine
- [ ] Streamlit frontend
- [ ] AWS deployment

### V2
- [ ] FHIR R4 standard integration (Epic/Cerner compatible)
- [ ] HL7 message parsing
- [ ] Multi-site trial filtering by geography
- [ ] REDCap integration for research sites
- [ ] HIPAA-compliant PHI handling
- [ ] Clinician dashboard with workflow integration

### V3
- [ ] Real-time trial status monitoring
- [ ] Patient consent workflow
- [ ] Outcome tracking
- [ ] IRB documentation generation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Built By

Cibi Siddarth — MS Computer Science, University of North Florida  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
