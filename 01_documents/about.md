# 📘 Module 01 — Documents

## 🎯 Goal

In this module, we learned the **foundation of every LangChain application**.

Everything in LangChain starts with a **Document**.

Before embeddings...
Before vector databases...
Before RAG...
Before LLM retrieval...

We first convert our raw data into **Document objects**.

---

# 📂 Module Structure

```
LangChain-Book/
│
├── 01_documents/
│   ├── 01_text_loader.py
│   ├── 02_document_object.py
│   ├── 03_manual_document.py
│   ├── 04_multiple_documents.py
│   ├── 05_pdf_loader.py
│   ├── 06_pdf_multiple_pages.py
│   ├── 07_csv_loader.py
│   ├── 08_csv_multiple_rows.py
│   ├── 09_directory_loader.py
│   └── 10_exercise.py
│
├── data/
│   ├── sample.txt
│   ├── sample.pdf
│   ├── employees.csv
│   └── documents/
│        ├── ai.txt
│        ├── ml.txt
│        ├── dl.txt
│        └── python.txt
│
└── .venv/
```

---

# 🚀 Before Starting

## Activate Virtual Environment

PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\.venv\Scripts\Activate.ps1
```

You should see

```
(.venv)
```

---

## Verify Python

```powershell
python --version
```

---

## Verify Installed Packages

```powershell
pip show langchain
pip show langchain-community
```

---

# What is a Document?

A Document is the basic data object used by LangChain.

It contains two parts:

```
Document
│
├── page_content
│
└── metadata
```

Example

```python
Document(
    page_content="Artificial Intelligence is transforming the world.",
    metadata={
        "source": "sample.txt"
    }
)
```

---

# Lesson 01 — TextLoader

## Learned

- Loading a text file
- Creating Document objects automatically

```python
loader = TextLoader(...)
documents = loader.load()
```

Result

```
Text File

↓

Document
```

---

# Lesson 02 — Understanding Document Object

Learned how to access

```
document.page_content

document.metadata
```

Example

```python
print(document.page_content)

print(document.metadata)
```

---

# Lesson 03 — Manual Document Creation

Created our own Document without loading any file.

```python
Document(
    page_content="Hello World",
    metadata={"author":"Raju"}
)
```

Important

A Document does NOT need a file.

It can come from

- Database
- API
- User Input
- Website
- Email
- Generated Text

---

# Lesson 04 — Multiple Documents

Created several Documents manually.

Stored them inside a list.

```
documents

├── Document 1

├── Document 2

└── Document 3
```

Learned iteration

```python
for doc in documents:
    print(doc.page_content)
```

---

# Lesson 05 — PDF Loader

Loaded

```
sample.pdf
```

using

```python
PyPDFLoader
```

Learned

A PDF is split into pages.

Each page becomes one Document.

```
PDF

↓

Page 1

↓

Document 1

Page 2

↓

Document 2
```

---

# Lesson 06 — Multiple PDF Pages

Iterated through every page.

Printed

- Content
- Metadata
- Page Number

Learned

```
metadata["page"]
```

---

# Lesson 07 — CSV Loader

Loaded

```
employees.csv
```

using

```python
CSVLoader
```

Each row became a Document.

```
CSV

↓

Row 1

↓

Document

Row 2

↓

Document

Row 3

↓

Document
```

Why?

Because Retrieval works better on smaller logical units.

---

# Lesson 08 — CSV Multiple Rows

Looped through every employee.

Printed

```
page_content

metadata
```

Learned

One employee = One Document

---

# Lesson 09 — Directory Loader

Loaded an entire folder.

Instead of

```
TextLoader()

TextLoader()

TextLoader()
```

we simply used

```python
DirectoryLoader(...)
```

Every file became one Document.

```
Folder

↓

File 1

↓

Document

File 2

↓

Document

File 3

↓

Document
```

---

# Lesson 10 — Final Exercise

Combined everything.

Learned to

- Load multiple files
- Iterate Documents
- Access metadata
- Count documents
- Count characters
- Print sources

---

# Important Classes Learned

| Class | Purpose |
|---------|----------|
| Document | Stores text + metadata |
| TextLoader | Loads text files |
| PyPDFLoader | Loads PDFs |
| CSVLoader | Loads CSV |
| DirectoryLoader | Loads multiple files |

---

# Important Properties

## page_content

Stores actual text.

Example

```python
document.page_content
```

---

## metadata

Stores information about the document.

Example

```python
document.metadata
```

Example metadata

```python
{
    "source":"sample.pdf",
    "page":2
}
```

---

# Mental Model

```
Raw Data

↓

Loader

↓

Document Objects

↓

Text Splitter

↓

Embeddings

↓

Vector Database

↓

Retriever

↓

LLM
```

Everything starts from **Documents**.

---

# Key Takeaways

✅ A Document is NOT a file.

It is one searchable piece of information.

---

A Document has

```
page_content

+

metadata
```

---

Different loaders create Documents differently.

| Loader | Creates |
|---------|----------|
| TextLoader | One Document |
| PDFLoader | One Document per page |
| CSVLoader | One Document per row |
| DirectoryLoader | One Document per file |

---

Metadata is extremely important.

Without metadata

- No source tracking
- No filtering
- No citations

---

Documents are the input to every RAG pipeline.

```
Documents

↓

Chunks

↓

Embeddings

↓

Vector Database

↓

Retrieval

↓

LLM
```

---

# Files Created

```
01_text_loader.py

02_document_object.py

03_manual_document.py

04_multiple_documents.py

05_pdf_loader.py

06_pdf_multiple_pages.py

07_csv_loader.py

08_csv_multiple_rows.py

09_directory_loader.py

10_exercise.py
```

---

# Module Status

```
Module 01

████████████████████

100% Complete ✅
```

---

# Next Module

```
02_text_splitters
```

We will answer

> Why do we split Documents into chunks before creating embeddings?

This is the beginning of the real RAG pipeline.