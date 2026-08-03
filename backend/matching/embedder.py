"""
TrialBridge — Day 8: BioBERT Embedding Pipeline
Generates semantic embeddings for all trials and stores them in pgvector.

Why embeddings? Plain keyword search fails for medical text.
"Cardiac failure" and "heart failure" mean the same thing but share no words.
BioBERT understands medical semantics — it was trained on 29M+ PubMed abstracts.
A patient with "cardiac failure" will match trials about "heart failure"
because their vectors are close in 768-dimensional space.
"""

import asyncio
import time

import asyncpg

DB_URL = "postgresql://trialbridge_user:devpassword@localhost:5432/trialbridge"

# We use the smaller but fast MiniLM model for V1
# It's 5x faster than BioBERT with 90% of the quality
# We'll upgrade to full BioBERT in Week 3
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32


def build_trial_text(trial: dict) -> str:
    """
    Build a rich text representation of a trial for embedding.
    The more context we give the model, the better the matches.
    We combine title + summary + conditions + eligibility snippet.
    """
    parts = []

    if trial.get("title"):
        parts.append(trial["title"])

    if trial.get("brief_summary"):
        # First 300 chars of summary — enough context without being too long
        parts.append(trial["brief_summary"][:300])

    if trial.get("conditions"):
        import json

        try:
            conditions = (
                json.loads(trial["conditions"])
                if isinstance(trial["conditions"], str)
                else trial["conditions"]
            )
            if conditions:
                parts.append("Conditions: " + ", ".join(conditions[:5]))
        except Exception:
            pass

    if trial.get("eligibility_criteria_raw"):
        # First 200 chars of eligibility — captures the key inclusion criteria
        parts.append(trial["eligibility_criteria_raw"][:200])

    return " | ".join(parts) if parts else trial.get("title", "")


async def embed_all_trials():
    """
    Main embedding pipeline:
    1. Load the sentence transformer model
    2. Pull all trials from PostgreSQL
    3. Generate embeddings in batches
    4. Store vectors back in PostgreSQL via pgvector
    """
    print("=" * 60)
    print("TrialBridge — Day 8: BioBERT Embedding Pipeline")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    print("(First run downloads ~90MB — subsequent runs are instant)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Connect to database
    print("\nConnecting to database...")
    conn = await asyncpg.connect(DB_URL)
    print("Connected")

    # Add embedding column if it doesn't exist
    print("\nSetting up embedding column...")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(f"""
        ALTER TABLE trials
        ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})
    """)
    await conn.execute("""
        ALTER TABLE trials
        ADD COLUMN IF NOT EXISTS embedding_text TEXT
    """)
    print("Embedding column ready")

    # Pull all trials
    print("\nFetching trials from database...")
    rows = await conn.fetch("""
        SELECT id, nct_id, title, brief_summary,
               conditions, eligibility_criteria_raw
        FROM trials
        ORDER BY nct_id
    """)
    total = len(rows)
    print(f"Found {total:,} trials to embed")

    # Generate embeddings in batches
    print(f"\nGenerating embeddings in batches of {BATCH_SIZE}...")
    print("This takes 2-5 minutes on CPU...\n")

    embedded = 0
    failed = 0
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        # Build text for each trial in batch
        texts = []
        for row in batch:
            text = build_trial_text(dict(row))
            texts.append(text)

        # Generate embeddings for entire batch at once
        # This is much faster than embedding one at a time
        try:
            embeddings = model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )

            # Store each embedding in the database
            for j, (row, embedding, text) in enumerate(zip(batch, embeddings, texts)):
                try:
                    await conn.execute(
                        """
                        UPDATE trials
                        SET embedding = $1::vector,
                            embedding_text = $2,
                            updated_at = NOW()
                        WHERE id = $3
                    """,
                        str(embedding.tolist()),
                        text[:500],
                        row["id"],
                    )
                    embedded += 1
                except Exception:
                    failed += 1

        except Exception as e:
            print(f"  Batch error: {e}")
            failed += len(batch)
            continue

        # Progress update every 5 batches
        if (i // BATCH_SIZE) % 5 == 0:
            elapsed = time.time() - start_time
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            print(
                f"  Progress: {embedded:,}/{total:,} embedded"
                f" | {rate:.0f} trials/sec"
                f" | ~{remaining:.0f}s remaining"
            )

    elapsed = time.time() - start_time

    # Create vector index for fast similarity search
    print("\nCreating vector similarity index...")
    print("(This makes search 100x faster than brute force)")
    try:
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trials_embedding
            ON trials USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10)
        """)
        print("Vector index created")
    except Exception as e:
        print(f"Index creation note: {e}")

    # Final stats
    embedded_count = await conn.fetchval("SELECT COUNT(*) FROM trials WHERE embedding IS NOT NULL")

    print("\n" + "=" * 60)
    print("Embedding pipeline complete!")
    print(f"Trials embedded     : {embedded:,}")
    print(f"Failed              : {failed}")
    print(f"Total in DB         : {embedded_count:,}")
    print(f"Time taken          : {elapsed:.1f}s")
    print(f"Rate                : {embedded / elapsed:.0f} trials/sec")
    print("=" * 60)

    # Test: find similar trials to a sample query
    print("\nTesting semantic search...")
    print("Query: 'Type 2 Diabetes treatment with medication'")

    query_embedding = model.encode(
        ["Type 2 Diabetes treatment with medication"],
        normalize_embeddings=True,
    )[0]

    similar = await conn.fetch(
        """
        SELECT nct_id, status,
               LEFT(title, 70) as short_title,
               1 - (embedding <=> $1::vector) as similarity
        FROM trials
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT 5
    """,
        str(query_embedding.tolist()),
    )

    print("\nTop 5 semantically similar trials:")
    for row in similar:
        print(f"  {row['nct_id']} | similarity: {row['similarity']:.3f} | {row['short_title']}")

    await conn.close()
    print("\nDay 8 complete. Semantic search is ready.")


if __name__ == "__main__":
    asyncio.run(embed_all_trials())
