"""
Lesson 01 - Loading a Text File using LangChain

Goal:
- Read a text file
- Convert it into LangChain Document objects
"""

# Import TextLoader
from langchain_community.document_loaders import TextLoader

# Path of the text file
FILE_PATH = "data/sample.txt"

# Create the loader object
loader = TextLoader(FILE_PATH)

# Load the file
# Returns a list of Document objects
documents = loader.load()

print("=" * 60)
print("Total Documents Loaded:", len(documents))
print("=" * 60)

# Print the first document
print(documents[0])

print("=" * 60)

# Check the object type
print("Type:", type(documents[0]))