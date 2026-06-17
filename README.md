# TrialBridge 🧬

> AI-powered clinical trial matching platform — connecting patients to eligible trials through intelligent EHR parsing, semantic search, and explainable matching.

[![CI](https://github.com/Cibisid/TRAILBRIDGE-/actions/workflows/ci.yml/badge.svg)](https://github.com/Cibisid/TRAILBRIDGE-/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## The Problem

Over 80% of clinical trials fail to meet enrollment targets. Simultaneously, millions of patients who qualify for trials that could save their lives never hear about them. The matching process today is done manually by research coordinators reading eligibility criteria line by line.

**TrialBridge solves this.**

---

## What It Does

A doctor pastes a patient note. TrialBridge reads it, understands it, and returns a ranked list of clinical trials the patient actually qualifies for — with a plain English explanation of why each trial matches or doesn't.

**Example:**

Input:
58-year-old female with Type 2 Diabetes. HbA1c 8.9%, eGFR 72.

Currently on Metformin. No prior insulin therapy.

Output (200 OK, 1ms):
```json
{
  "age": 58,
  "sex": "female",
  "primary_diagnosis": "Type 2 Diabetes",
  "current_medications": ["Metformin"],
  "negated_conditions": ["insulin therapy"],
  "lab_values": {"HbA1c": 8.9, "eGFR": 72},
  "extraction_confidence": 1.0,
  "query_string": "58 year old patient. female. diagnosed with Type 2 Diabetes. currently taking Metformin. lab values: HbA1c 8.9, eGFR 72. no history of insulin therapy."
}
```

---

## Build Status — Day 7 of 30

| Component | Status | Details |
|---|---|---|
| Production Infrastructure | ✅ Complete | Docker, FastAPI, PostgreSQL+pgvector, Redis, Nginx, CI/CD |
| Trial Database | ✅ Complete | 2,344 real trials from ClinicalTrials.gov bulk export |
| Data Quality Report | ✅ Complete | 103,509 XML files analyzed, NLP blueprint created |
| NLP Extraction Pipeline | ✅ Complete | Age, sex, diagnosis, labs, medications, negation |
| Bug Fixes — V2 Extractor | ✅ Complete | Negation-aware diagnosis, duplicate deduplication |
| Live API Endpoint | ✅ Complete | POST /api/v1/parse-patient — 1ms response time |
| Semantic Search (BioBERT) | 🔄 Week 2 | pgvector similarity search |
| Eligibility Rules Engine | 🔄 Week 2 | Age, conditions, lab value parsing |
| Explanation Engine | 🔄 Week 3 | Claude API integration |
| Streamlit Frontend | 🔄 Week 3 | Live demo UI |
| AWS Deployment | 🔄 Week 4 | EC2, RDS, ElastiCache |

---

## Architecture
┌──────────────────────────────────────────────────────────────┐

│                         TrialBridge                          │

│                                                              │

│   ┌─────────────┐         ┌──────────────────┐              │

│   │  EHR Note   │──────▶  │  NLP Pipeline    │  ✅ LIVE     │

│   │  (raw text) │         │  regex + custom  │              │

│   └─────────────┘         └────────┬─────────┘              │

│                                    │                         │

│                           ┌────────▼─────────┐              │

│                           │  Match Engine    │  🔄 Week 2   │

│                           │  BioBERT+pgvector│              │

│                           └────────┬─────────┘              │

│                                    │                         │

│                           ┌────────▼─────────┐              │

│                           │  Rules Engine    │  🔄 Week 2   │

│                           │  eligibility     │              │

│                           └────────┬─────────┘              │

│                                    │                         │

│                           ┌────────▼─────────┐              │

│                           │  Explanation     │  🔄 Week 3   │

│                           │  Claude API      │              │

│                           └────────┬─────────┘              │

│                                    │                         │

│   ✅ PostgreSQL + pgvector ◀───────┘                        │

│   ✅ Redis cache                                             │

│   ✅ FastAPI + JWT auth                                      │

│   ✅ Prometheus + Grafana                                    │

└──────────────────────────────────────────────────────────────┘

---

## Tech Stack

| Layer | Technology | Status |
|---|---|---|
| API Framework | FastAPI | ✅ Running |
| Database | PostgreSQL + pgvector | ✅ Running |
| Cache | Redis | ✅ Configured |
| Task Queue | Celery | ✅ Configured |
| Medical NER | Custom regex extractors | ✅ Running |
| Embeddings | BioBERT (HuggingFace) | 🔄 Week 2 |
| Vector Search | pgvector | 🔄 Week 2 |
| Explanation | Claude API (Anthropic) | 🔄 Week 3 |
| Frontend | Streamlit | 🔄 Week 3 |
| Infrastructure | Docker + Docker Compose | ✅ Running |
| Cloud | AWS (EC2, RDS, S3, ElastiCache) | 🔄 Week 4 |
| CI/CD | GitHub Actions | ✅ Running |
| Monitoring | Prometheus + Grafana + Sentry | ✅ Configured |
| Reverse Proxy | Nginx | ✅ Configured |

---

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/Cibisid/TRAILBRIDGE-.git
cd TRAILBRIDGE-
```

### 2. Set up environment variables
```bash
cp .env.example .env
```

### 3. Start PostgreSQL
```bash
docker compose up postgres -d
```

### 4. Start the API server
```bash
python -m uvicorn backend.main:app --reload
```

### 5. Open API docs
Visit **http://127.0.0.1:8000/docs**

Try the `POST /api/v1/parse-patient` endpoint with this note:
```json
{
  "note": "58-year-old female with Type 2 Diabetes. HbA1c 8.9%, eGFR 72. Currently on Metformin. No prior insulin therapy. No history of cardiovascular disease."
}
```

---

## Project Structure
trialbridge/

├── backend/

│   ├── api/

│   │   └── v1/

│   │       └── endpoints/

│   │           ├── auth.py         # JWT authentication

│   │           ├── health.py       # Health checks

│   │           ├── match.py        # Matching engine (Week 2)

│   │           ├── patient.py      # NLP extraction endpoint

│   │           └── trials.py       # Trial listing (Week 2)

│   ├── core/

│   │   ├── config.py               # Pydantic settings

│   │   ├── database.py             # Async SQLAlchemy + pgvector

│   │   ├── logging.py              # Structured JSON logging

│   │   └── security.py             # JWT + bcrypt

│   ├── models/

│   │   └── init.py             # Trial, PatientProfile, MatchResult, AuditLog

│   ├── nlp/

│   │   ├── extractor.py            # V1 NLP pipeline (Day 4)

│   │   └── extractor_v2.py         # V2 with bug fixes (Day 5) ✅ ACTIVE

│   ├── matching/                   # Semantic search (Week 2)

│   └── tests/

│       └── test_health.py

├── scripts/

│   ├── ingest_trials.py            # ClinicalTrials.gov ingestion

│   └── explore_data.py             # Data quality analysis

├── frontend/                       # Streamlit UI (Week 3)

├── infra/

│   ├── docker/                     # Dockerfiles + init scripts

│   └── nginx/                      # Reverse proxy config

├── reports/

│   └── day3_data_quality_report.txt

├── .github/

│   └── workflows/

│       └── ci.yml                  # Lint → Test → Build

├── docker-compose.yml

├── .env.example

└── README.md

---

## Live API Endpoints

| Method | Endpoint | Status | Description |
|---|---|---|---|
| GET | `/health` | ✅ Live | Liveness check |
| GET | `/health/ready` | ✅ Live | Readiness + DB check |
| GET | `/metrics` | ✅ Live | Prometheus metrics |
| POST | `/api/v1/auth/token` | ✅ Live | Get JWT token |
| POST | `/api/v1/parse-patient` | ✅ Live | EHR note → structured profile |
| POST | `/api/v1/match` | 🔄 Week 2 | Patient → ranked trial matches |
| GET | `/api/v1/trials` | 🔄 Week 2 | List and search trials |

---

## NLP Extraction Capabilities

The `POST /api/v1/parse-patient` endpoint currently extracts:

| Field | Example Input | Example Output |
|---|---|---|
| Age | "58-year-old", "58 y/o", "age 58" | 58 |
| Sex | "female", "she", "F/58" | "female" |
| Primary diagnosis | "Type 2 Diabetes Mellitus" | "Type 2 Diabetes Mellitus" |
| Comorbidities | "also has hypertension" | ["Hypertension"] |
| Negated conditions | "no history of stroke" | ["stroke"] |
| Current medications | "currently on Metformin" | ["Metformin"] |
| Prior treatments | "previously treated with Carboplatin" | ["Carboplatin"] |
| Lab values | "HbA1c 8.9%, eGFR 72" | {"HbA1c": 8.9, "eGFR": 72} |
| Confidence score | — | 0.0 – 1.0 |

---

## Data

| Property | Value |
|---|---|
| Source | ClinicalTrials.gov official bulk XML export |
| Total available | 103,509 trials |
| Currently loaded | 2,344 trials |
| Fields stored | NCT ID, title, eligibility criteria, age ranges, conditions, interventions, sponsor, locations, phase, status |
| Target (Week 2) | 50,000+ trials with BioBERT embeddings |

---

## Week 2 Roadmap (Days 8–14)

- **Day 8:** Install sentence-transformers, embed all 2,344 trials with BioBERT, store vectors in pgvector
- **Day 9:** Semantic search — patient query string → cosine similarity → top 50 candidates
- **Day 10:** Eligibility rules engine V1 — age range, gender, healthy volunteer filters
- **Day 11:** Advanced rules — lab value thresholds, negation constraints, temporal parsing
- **Day 12:** Hybrid scoring — semantic score + eligibility score = composite match score
- **Day 13:** POST /api/v1/match endpoint live and tested
- **Day 14:** End-to-end demo — paste a note, get ranked trials with scores

---

## Why This Matters

Clinical trial matching is a $2B+ problem in the healthcare industry. Companies like Antidote Health, TrialSpark, and IQVIA employ hundreds of people to do this manually. TrialBridge is building the infrastructure to automate it.

By Day 30, any hospital system will be able to integrate with TrialBridge via a single API call.

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE)

---

## Built By

**Cibi Siddarth**
MS Computer Science (Data Science Concentration)
University of North Florida — GPA 3.9

Building TrialBridge in public for 30 days.

[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/Cibisid)