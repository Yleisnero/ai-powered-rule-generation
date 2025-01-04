import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional, Union
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS

from rag import get_documents_embeddings
from embeddings import bge_small_en_v15_embeddings

class ReactionTestSpecs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reaction_signal: Optional[str] = ""
    window_shift_event_start: Optional[Union[int, float]] = None
    window_shift_event_end: Optional[Union[int, float]] = None
    feature: Optional[str] = ""
    kind: Optional[str] = ""
    threshold: Optional[Union[int, float, str]] = None
    second_threshold: Optional[Union[int, float, str]] = None
    accepted_ratio_invalid: Optional[Union[int, float]] = None


class SingleSignalTestSpecs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Optional[str] = ""
    threshold: Union[int, float, str] = None
    second_threshold: Optional[Union[int, float, str]] = None
    stat_feature: Optional[str] = ""
    window_length: Optional[Union[int, float]] = None
    min_periods: Optional[Union[int, float]] = None


class SimilarityTestSpecs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Optional[str] = ""
    threshold: Optional[Union[int, float, str]] = None
    second_threshold: Optional[Union[int, float, str]] = None
    stat_feature: Optional[str] = ""
    window_length: Optional[Union[int, float]] = None
    min_periods: Optional[Union[int, float]] = None
    # test_signals: Optional[List[Any]]


class Rule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    test_method: Optional[str]
    test_preconditions: Optional[dict] = None
    time_start_utc: Optional[str]
    time_end_utc: Optional[str]
    violation_plausibility_code: Optional[str] = ""


class ReactionRule(Rule):
    model_config = ConfigDict(extra="ignore")

    test_specs: Optional[ReactionTestSpecs] = None
    event_dict: Optional[dict] = None


class SingleSignalRule(Rule):
    model_config = ConfigDict(extra="ignore")

    test_method_sub: Optional[str] = None
    test_virtual_variables: Optional[Dict[Any, Any]] = None
    event_dict: Optional[Dict[Any, Any]] = None
    test_specs: Optional[SingleSignalTestSpecs] = None
    violation_agg_func_runs: Optional[str] = ""


class SimilarityRule(Rule):
    model_config = ConfigDict(extra="ignore")

    test_specs: Optional[SimilarityTestSpecs] = None
    violation_agg_func_runs: Optional[str] = ""


ReactionTestSpecs.model_rebuild()
SimilarityTestSpecs.model_rebuild()
SingleSignalTestSpecs.model_rebuild()
SimilarityRule.model_rebuild()
SingleSignalRule.model_rebuild()
ReactionRule.model_rebuild()
Rule.model_rebuild()


def read_examples(example_path="../examples") -> list:
    examples = []
    for file in os.listdir(example_path):
        if file.endswith(".txt"):
            with open(os.path.join(example_path, file), "r", encoding="utf-8") as f:
                examples.append(json.loads(f.read()))
    return examples


class DocumentRAGTool(BaseTool):
    name: str = "Document RAG Tool"
    description: str = (
        "Useful when you want to retrieve more contextual information for a query. Pass the query as a single string argument to use this tool."
    )

    def _run(self, query: str) -> str:
        return get_documents_embeddings(query, 2)


# document_rag_tool = DocumentRAGTool()

load_dotenv()

mistral_llm = ChatOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL"),
    model=os.environ.get("API_MODEL"),
    temperature=0,
)


def setup_crew(query, rule_template, context, examples, rule_type) -> Crew:
    rulecreator = Agent(
        role="Rule Creator",
        goal="Your goal is to create a integrity check rule. This rule ensures that a specific function of the building operates correctly. You will be given a standardized template of the rule alongside which what specific function should be tested.",
        backstory="You are an expert in building automation, with knowledge of Heating, Ventilation, and Air Conditioning (HVAC) systems as well as Air Handling Units (AHUs). You have extensive experience in creating rules for integrity checks in buildings. Over the years, you have gained substantial experience in formulating effective rules for conducting integrity checks in buildings.",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=mistral_llm,
    )

    validator = Agent(
        role="Rule Validator",
        goal="Your goal is to check if a rule fullfills the syntactic requirements of the query. You will be provided with a rule and an explanation and have to output it in the correct format.",
        backstory="You are an expert in building automation, with knowledge of Heating, Ventilation, and Air Conditioning (HVAC) systems as well as Air Handling Units (AHUs). You have extensive experience in validating rules for integrity checks in buildings. Over the years, you have gained substantial experience in providing integrity checks in the correct format.",
        verbose=True,
        allow_delegation=False,
        tools=[],
        llm=mistral_llm,
    )

    task1 = Task(
        description=f"You are provided with the rule template: \n{rule_template}\n Your specific task is to identify an appropriate rule that satisfy the query: {query}\n Your objective is to analyze the given information, apply your understanding, and determine a rule that aligns with the requirements specified in {query}. Context: {context} \n Examples: {examples}",
        expected_output=f"The final result should be a rule following the rule template and satisfying the query with explanation why this rule is fitting the query.",
        agent=rulecreator,
    )

    task2 = Task(
        description=f"You are provided with a rule and an explanation. Your task is to output only the json in correct format.",
        expected_output=f"The final result should be a rule in correct json format. The output should be a raw json object.",
        agent=validator,
        output_pydantic=rule_type,
    )

    crew = Crew(
        agents=[rulecreator, validator],
        tasks=[task1, task2],
        verbose=2,
    )

    return crew


def run_crew(query: str, examples: list, empty_rule: Rule, rule_type) -> Rule:
    print("📝 Retrieving context ...")
    context = get_documents_embeddings(query, k=3)

    rule_template = empty_rule.model_dump_json()

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Question: {input}\nAnswer: {output}",
    )

    print("📋 Picking best examples ...")
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # The list of examples available to select from.
        examples,
        # The embedding class used to produce embeddings which are used to measure semantic similarity.
        bge_small_en_v15_embeddings(),
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        FAISS,
        # The number of examples to produce.
        k=4,
    )

    similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="",
        suffix="Question: {query}\nAnswer:",
        input_variables=["query"],
    )

    examples = similar_prompt.format(query=query)

    crew = setup_crew(query, rule_template, context, examples, rule_type)
    result = crew.kickoff()
    return result
