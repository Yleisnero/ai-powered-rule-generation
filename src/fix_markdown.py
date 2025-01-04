from prompt_llm import prompt_mistral


def fix_markdown_table(markdown_table: str) -> str:
    markdown_table = markdown_table.replace("|", " ").replace("\n", " ")
    input = f"""
    The following information was taken from a markdown table. Please use the information and put it in a markdown table format.
    Information:
    {markdown_table}"""
    result = prompt_mistral(input)
    return result
