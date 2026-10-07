"""
01_pgvector_architecture.py

Title:
    PGVector Architecture Overview

Goal:
    Understand how the pgvector extension integrates into PostgreSQL engine,
    and inspect the internal table schemas used by LangChain PGVector.

Architecture Highlights:
    1. Postgres Extension: `pgvector` adds the native `vector` data type directly
       into PostgreSQL's type system.
    2. Relational & Vector Storage: Embeddings sit alongside relational columns in standard tables.
    3. LangChain Schema:
       - `langchain_pg_collection`: Stores collections (namespaces for documents).
       - `langchain_pg_embedding`: Stores actual vectors, text chunks (`cstore`),
         and JSONB metadata (`cmetadata`).
"""

import psycopg

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
DB_URL = "postgresql://langchain:langchain@localhost:6024/langchain"


def inspect_pgvector_architecture():
    print("=" * 70)
    print("1. CONNECTING TO POSTGRESQL & INSPECTING ARCHITECTURE")
    print("=" * 70)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                # ---------------------------------------------------------
                # Step 1: Verify pgvector extension status
                # ---------------------------------------------------------
                print("\n[Step 1] Checking Installed PostgreSQL Extensions:")
                cur.execute("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname = 'vector';
                """)
                ext = cur.fetchone()
                if ext:
                    print(f"  [OK] Extension '{ext[0]}' is installed (Version: {ext[1]})")
                else:
                    print("  [INFO] Extension 'vector' is NOT installed.")

                # ---------------------------------------------------------
                # Step 2: Inspect LangChain PGVector Tables
                # ---------------------------------------------------------
                print("\n[Step 2] Inspecting LangChain Collection & Embedding Tables:")
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name LIKE 'langchain_pg_%';
                """)
                tables = [row[0] for row in cur.fetchall()]
                print(f"  Found LangChain Tables: {tables}")

                # ---------------------------------------------------------
                # Step 3: Column Details for langchain_pg_embedding
                # ---------------------------------------------------------
                print("\n[Step 3] Columns of 'langchain_pg_embedding' Table:")
                cur.execute("""
                    SELECT column_name, data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'langchain_pg_embedding'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                for col_name, data_type, udt_name in columns:
                    type_str = udt_name if data_type == 'USER-DEFINED' else data_type
                    print(f"  - {col_name:<20} | Type: {type_str}")

                # ---------------------------------------------------------
                # Step 4: Summary of Architectural Advantages
                # ---------------------------------------------------------
                print("\n" + "=" * 70)
                print("ARCHITECTURAL SUMMARY")
                print("=" * 70)
                print("  * Unified Storage: Embeddings, text chunks, and metadata live in Postgres.")
                print("  * ACID Guarantees: Full support for transactions, rollbacks, and crash recovery.")
                print("  * JSONB Filtering: Metadata is stored as JSONB for fast indexing & filtering.")
                print("  * Operator Overloading: Custom operators (<=>, <->, <#>) handle vector distance.")
                print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Failed to connect to PostgreSQL: {e}")



def main():
    inspect_pgvector_architecture()


if __name__ == "__main__":
    main()

