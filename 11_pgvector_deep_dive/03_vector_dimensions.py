"""
03_vector_dimensions.py

Title:
    Vector Dimensions & Type Constraints in PGVector

Goal:
    Inspect vector dimensions produced by embedding models (`nomic-embed-text`),
    define explicit fixed-dimension vector tables in PostgreSQL (`vector(N)`),
    and observe how PostgreSQL enforces vector dimension validation on insertion.
"""

import psycopg
from langchain_ollama import OllamaEmbeddings

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
DB_URL = "postgresql://langchain:langchain@localhost:6024/langchain"
EMBEDDING_MODEL = "nomic-embed-text"


def demonstrate_vector_dimensions():
    print("=" * 70)
    print("3. VECTOR DIMENSIONS & TYPE CONSTRAINTS DEMO")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Generate Embedding & Inspect Dimensionality
    # ---------------------------------------------------------
    print("\n[Step 1] Embedding text with OllamaEmbeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    sample_text = "PostgreSQL pgvector extension for high-performance vector search."
    vector = embeddings.embed_query(sample_text)
    dim_count = len(vector)

    print(f"  Sample Text: '{sample_text}'")
    print(f"  [OK] Model Output Dimension Count: {dim_count} dimensions")
    print(f"  First 5 floats: {vector[:5]}")

    # ---------------------------------------------------------
    # Step 2: Create Vector Table with Explicit Dimensions
    # ---------------------------------------------------------
    print("\n[Step 2] Creating Temporary Postgres Table with `vector(768)` column:")
    table_name = "test_vector_dimensions_demo"

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                cur.execute(f"""
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        embedding vector({dim_count})
                    );
                """)
                conn.commit()
                print(f"  [OK] Table '{table_name}' created with column `embedding vector({dim_count})`.")

                # ---------------------------------------------------------
                # Step 3: Insert Valid Embedding Vector
                # ---------------------------------------------------------
                print("\n[Step 3] Inserting valid vector of size 768:")
                vector_str = f"[{','.join(map(str, vector))}]"
                cur.execute(f"""
                    INSERT INTO {table_name} (content, embedding)
                    VALUES (%s, %s::vector);
                """, (sample_text, vector_str))
                conn.commit()
                print("  [OK] Successfully inserted 768-dimensional vector into database.")

                # ---------------------------------------------------------
                # Step 4: Demonstrate Mismatch Error Handling
                # ---------------------------------------------------------
                print("\n[Step 4] Attempting to insert mismatched vector (size = 3):")
                mismatched_vector_str = "[1.0, 2.0, 3.0]"
                try:
                    cur.execute(f"""
                        INSERT INTO {table_name} (content, embedding)
                        VALUES (%s, %s::vector);
                    """, ("Mismatched Vector Test", mismatched_vector_str))
                    conn.commit()
                except psycopg.errors.DataException as err:
                    conn.rollback()
                    print("  [OK] Expected Dimension Mismatch Error Caught!")
                    print(f"    PostgreSQL Response: {err}".strip())

                # ---------------------------------------------------------
                # Step 5: Clean Up
                # ---------------------------------------------------------
                print("\n[Step 5] Cleaning up temporary table:")
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                conn.commit()
                print(f"  [OK] Table '{table_name}' dropped.")

                print("\n" + "=" * 70)
                print("[OK] Vector dimension verification and enforcement demo complete!")
                print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Failed database execution: {e}")



def main():
    demonstrate_vector_dimensions()


if __name__ == "__main__":
    main()

