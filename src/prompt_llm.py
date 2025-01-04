import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

def prompt_mistral(input: str):
    load_dotenv()

    llm = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("API_BASE_URL"),
        model=os.environ.get("API_MODEL"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.invoke(input)


def prompt_mistral_pydantic(input: str, model: BaseModel):
    load_dotenv()

    llm = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("API_BASE_URL"),
        model=os.environ.get("API_MODEL"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    parser = PydanticOutputParser(pydantic_object=model)

    chain = prompt | llm | parser

    return chain.invoke(input)
