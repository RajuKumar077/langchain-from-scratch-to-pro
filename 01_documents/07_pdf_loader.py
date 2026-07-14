"""
Lesson 07 - Loading a PDF

Goal:
- Load a PDF
- Understand that every page becomes a Document
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# ==========================================================
# PDF Path
# ==========================================================

BASE_DIR = Path(__file__).parent.parent
PDF_PATH = BASE_DIR / "data" / "Raju_Kumar_AI_ML_Engineer.pdf"

# ==========================================================
# Create Loader
# ==========================================================

loader = PyPDFLoader(PDF_PATH)

print("=" * 60)
print("PDF Loader Created")
print("=" * 60)

# ==========================================================
# Load PDF
# ==========================================================

documents = loader.load()

print("\n" + "=" * 60)
print("Total Documents")
print("=" * 60)

print(len(documents))

# ==========================================================
# Print Every Page
# ==========================================================

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 60)
    print(f"Page {index}")
    print("=" * 60)

    print(document.page_content[:500])

    print("\nMetadata")
    print(document.metadata)