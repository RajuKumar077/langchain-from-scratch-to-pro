"""
Lesson 08 - CSV Loader

Goal:
- Load a CSV file
- Every row becomes one Document
"""

from pathlib import Path

from langchain_community.document_loaders.csv_loader import CSVLoader

# ==========================================================
# CSV Path
# ==========================================================

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "data" / "employees.csv"

# ==========================================================
# Create Loader
# ==========================================================

loader = CSVLoader(file_path=CSV_PATH)

print("=" * 60)
print("CSV Loader Created")
print("=" * 60)

# ==========================================================
# Load CSV
# ==========================================================

documents = loader.load()

print("\n" + "=" * 60)
print("Total Documents")
print("=" * 60)

print(len(documents))

# ==========================================================
# Print Every Document
# ==========================================================

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 60)
    print(f"Row {index}")
    print("=" * 60)

    print("Page Content")
    print(document.page_content)

    print("\nMetadata")
    print(document.metadata)


print("\nFirst Employee")
print(documents[0].page_content)

print("\nSecond Employee")
print(documents[1].page_content)