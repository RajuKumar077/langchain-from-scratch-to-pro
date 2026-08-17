# user input to make use of pydentic parser with ollama given by porffesion 



from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an AI assistant that extracts
        structured information from text.
        """
    ),
    (
        "human",
        """
        Create a fictional professional profile
        based on this request:

        {person}
        """
    ),
])
class Person(BaseModel):

    name: str = Field(
        description="The person's name"
    )

    role: str = Field(
        description="The person's job role"
    )

    skills: list[str] = Field(
        description="The person's technical skills"
    )
llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)

try:
    person  =input("Enter the professionengineeal you want: ")
    structured_llm = llm.with_structured_output(Person)
    chain = prompt | structured_llm
    response = chain.invoke({"person": person})
    print("Name   :", response.name)
    print("Role   :", response.role)
    print("Skills :", response.skills)
except Exception as e:
    print(e)
