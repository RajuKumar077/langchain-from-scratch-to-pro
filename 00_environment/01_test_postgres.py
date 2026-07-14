from sqlalchemy import create_engine, text

# PostgreSQL Connection String
DATABASE_URL = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

print("Creating Engine...")

engine = create_engine(DATABASE_URL)

print("Connecting to PostgreSQL...")

with engine.connect() as conn:

    print("Connected Successfully!")

    result = conn.execute(text("SELECT version();"))

    print(result.fetchone()[0])