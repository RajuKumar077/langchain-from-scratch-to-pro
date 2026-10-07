"""
02_postgres_vector_extension.py

Title:
    PostgreSQL Vector Extension Setup & Operator Inspection

Goal:
    Learn how to enable the `vector` extension, inspect registered vector operator classes,
    and execute raw SQL vector operations (casting, vector math, distance operators).

Key Operators:
    - `<=>` : Cosine Distance
    - `<->` : Euclidean / L2 Distance
    - `<#>` : Negative Inner Product
"""

import psycopg

DB_URL = "postgresql://langchain:langchain@localhost:6024/langchain"


def demonstrate_vector_extension():
    print("=" * 70)
    print("2. POSTGRESQL VECTOR EXTENSION & OPERATOR DEMO")
    print("=" * 70)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                # ---------------------------------------------------------
                # Step 1: Enable extension dynamically
                # ---------------------------------------------------------
                print("\n[Step 1] Enabling vector extension (CREATE EXTENSION IF NOT EXISTS vector):")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                print("  [OK] Extension 'vector' enabled successfully.")

                # ---------------------------------------------------------
                # Step 2: Query vector operator classes
                # ---------------------------------------------------------
                print("\n[Step 2] Registered Operator Classes for 'vector':")
                cur.execute("""
                    SELECT opcname 
                    FROM pg_opclass 
                    WHERE opcname LIKE 'vector%';
                """)
                opclasses = [row[0] for row in cur.fetchall()]
                for opc in opclasses:
                    print(f"  - Operator Class: {opc}")

                # ---------------------------------------------------------
                # Step 3: Raw SQL Vector Type Casting & Operators
                # ---------------------------------------------------------
                print("\n[Step 3] Raw SQL Distance Calculations:")
                v1 = "[1.0, 2.0, 3.0]"
                v2 = "[4.0, 5.0, 6.0]"

                cur.execute("""
                    SELECT 
                        %s::vector <=> %s::vector AS cosine_distance,
                        %s::vector <-> %s::vector AS euclidean_distance,
                        %s::vector <#> %s::vector AS inner_product;
                """, (v1, v2, v1, v2, v1, v2))

                cosine_dist, l2_dist, inner_prod = cur.fetchone()
                print(f"  Vector 1: {v1}")
                print(f"  Vector 2: {v2}")
                print(f"  - Cosine Distance (<=>):        {cosine_dist:.6f}")
                print(f"  - Euclidean Distance (<->):     {l2_dist:.6f}")
                print(f"  - Negative Inner Product (<#>): {inner_prod:.6f}")

                # ---------------------------------------------------------
                # Step 4: Vector Addition & Scalar Operations
                # ---------------------------------------------------------
                print("\n[Step 4] Vector Arithmetic Operators in SQL:")
                cur.execute("""
                    SELECT 
                        %s::vector + %s::vector AS sum_vec,
                        %s::vector - %s::vector AS diff_vec;
                """, (v1, v2, v1, v2))
                vec_sum, vec_diff = cur.fetchone()
                print(f"  - Sum (v1 + v2):  {vec_sum}")
                print(f"  - Diff (v1 - v2): {vec_diff}")

                print("\n" + "=" * 70)
                print("[OK] PostgreSQL vector extension ready for application usage!")
                print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Error testing vector extension: {e}")



def main():
    demonstrate_vector_extension()


if __name__ == "__main__":
    main()

