from langchain_text_splitters import MarkdownHeaderTextSplitter


def split_on_markdown(path):
    with open(path) as f:
        full_text = f.read()

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
        return_each_line=False,
    )

    texts = markdown_splitter.split_text(full_text)
    return texts
