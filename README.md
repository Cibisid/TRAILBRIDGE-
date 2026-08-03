# TrialBridge

AI-powered clinical trial matching. Parses free-text patient notes, searches trials by meaning
rather than keywords, applies eligibility rules, and explains every match in plain English.

[![CI](https://github.com/Cibisid/trialbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/Cibisid/trialbridge/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The problem

Over 80% of clinical trials miss their enrollment targets. At the same time, patients who would
qualify for a trial never hear about it. Today the matching is done by hand: a research
coordinator reads eligibility criteria line by line against a patient chart.

## What it does

A clinician pastes a patient note. TrialBridge returns a ranked list of trials the patient
actually qualifies for, each with a plain-English explanation of why it matches or doesn't.

**Input**

```
58-year-old female with Type 2 Diabetes. HbA1c 8.9%, eGFR 72.
Currently on Metformin. No prior insulin therapy.
```

**Output** — `POST /api/v1/parse-patient`, ~1 ms

```json
{
  "age": 58,
  "sex": "female",
  "primary_diagnosis": "Type 2 Diabetes",
  "current_medications": ["Metformin"],
  "negated_conditions": ["insulin therapy"],
  "lab_values": { "HbA1c": 8.9, "eGFR": 72 },
  "extraction_confidence": 1.0
}
```

`POST /api/v1/match` runs the same note through the full pipeline and returns scored,
explained trial matches.

---

## Pipeline

```
   Patient note (raw text)
            │
            ▼
   NLP extraction              backend/nlp/extractor_v2.py
   age · sex · diagnosis · labs · medications · negation
            │
            ▼
   Sentence embedding          backend/matching/embedder.py
   MiniLM → 384-dim vector
            │
            ▼
   Vector search               PostgreSQL + pgvector
   cosine similarity → top 50 candidates
            │
            ▼
   Eligibility rules           backend/matching/matcher.py
   age ranges · required + excluded conditions · lab thresholds
            │
            ▼
   Composite scoring → ranked matches
            │
            ▼
   Explanation                 backend/matching/explainer.py
   Claude API → plain-English rationale per match
```

---

## What's built

| Component | Status | Notes |
|---|---|---|
| Infrastructure | Done | Docker, docker-compose, FastAPI, PostgreSQL + pgvector, Redis, Nginx |
| CI/CD | Done | GitHub Actions: ruff lint, ruff format, mypy, pytest + coverage against live Postgres and Redis services |
| Authentication | Done | JWT bearer tokens, bcrypt hashing, stateless validation |
| Trial corpus | Done | 2,344 trials ingested from the ClinicalTrials.gov bulk export |
| Data quality analysis | Done | 103,509 source XML files parsed; findings drove the extraction rules |
| NLP extraction | Done | Age, sex, diagnosis, labs, medications; negation-aware, deduplicated |
| `POST /api/v1/parse-patient` | Done | ~1 ms response |
| Semantic search | Done | sentence-transformers MiniLM embeddings stored in pgvector |
| Eligibility rules engine | Done | Age, required/excluded conditions, lab-value thresholds |
| Composite match scoring | Done | Semantic similarity blended with rule satisfaction |
| `POST /api/v1/match` | Done | Full pipeline end to end |
| Claude explanation engine | Done | Per-match rationale via the Anthropic API |
| Streamlit demo UI | In progress | `frontend/app.py` |
| AWS deployment | Planned | EC2, RDS, ElastiCache |

**On the embedding model:** the original plan was BioBERT. V1 ships
`sentence-transformers/all-MiniLM-L6-v2` instead — roughly 5x faster inference at close to the
same retrieval quality on this corpus. BioBERT remains a swap-in option; the embedder is
model-agnostic.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI, async SQLAlchemy, Pydantic |
| Database | PostgreSQL 16 + pgvector |
| Cache / queue | Redis, Celery |
| Embeddings | sentence-transformers (MiniLM) |
| Explanations | Anthropic Claude API |
| Auth | JWT (python-jose), bcrypt |
| Infrastructure | Docker multi-stage builds, docker-compose, Nginx |
| CI | GitHub Actions — ruff, mypy, pytest, coverage |
| Monitoring | Prometheus |

---

## Running it

**Requires:** Docker, Python 3.11+

```bash
git clone https://github.com/Cibisid/trialbridge.git
cd trialbridge

cp .env.example .env          # add your ANTHROPIC_API_KEY for explanations

docker compose up postgres redis -d
pip install -r backend/requirements.txt

python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /api/v1/parse-patient`:

```json
{ "note": "58-year-old female with Type 2 Diabetes. HbA1c 8.9%, eGFR 72. Currently on Metformin. No prior insulin therapy." }
```

To load the trial corpus and build embeddings:

```bash
python scripts/ingest_trials.py
python -m backend.matching.embedder
```

---

## Layout

```
trialbridge/
├── backend/
│   ├── api/v1/endpoints/     auth · health · match · patient · trials
│   ├── core/                 config · database · logging · security
│   ├── matching/             embedder · matcher · explainer
│   ├── nlp/                  extractor · extractor_v2
│   └── tests/
├── frontend/                 Streamlit demo
├── infra/
│   ├── docker/               backend + frontend Dockerfiles, postgres init, prometheus
│   └── nginx/
├── scripts/                  ingest_trials · explore_data
└── .github/workflows/ci.yml
```

---

## Design notes

**Why vector search and not keyword search.** "Cardiac failure" and "heart failure" mean the same
thing and share no words. Embeddings put them close in vector space; a keyword index puts them
infinitely far apart.

**Why rules on top of vectors.** Semantic similarity finds *relevant* trials. It does not know
that a 58-year-old is excluded from an 18–45 study. Retrieval is fuzzy; eligibility is not. The
rules engine is deterministic and runs after retrieval narrows the field.

**Why JWT.** Stateless — the server stores no sessions, so any instance can validate any token.
That scales horizontally without a shared session store.

**Why negation handling.** "No prior insulin therapy" and "on insulin therapy" are opposite facts
that differ by one word. An extractor that misses negation will confidently produce the wrong
patient profile, and every downstream match inherits the error.

---

## License

MIT
