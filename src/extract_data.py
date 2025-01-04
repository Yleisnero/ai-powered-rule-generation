import shutil
import pymupdf4llm
import pymupdf
import os
from pathlib import Path

from fix_markdown import fix_markdown_table


def prepare_dir(file):
    path = f"../doc/{Path(file).stem}"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(f"../doc/{Path(file).stem}")
    with open(
        f"../doc/{Path(file).stem}/{Path(file).stem}.md", "w", encoding="utf-8"
    ) as f:
        f.write("")


def extract_pdf_data(file):
    prepare_dir(file)
    pdf_file = pymupdf.open(file)
    base_path = Path(file).stem
    for page_index in range(len(pdf_file)):
        print(f"Processing page {page_index}...")
        md_text = extract_markdown(file, page_index)
        table_paths = extract_tables(base_path, pdf_file, page_index)
        with open(f"../doc/{base_path}/{base_path}.md", "a", encoding="utf-8") as f:
            f.write(md_text)

            if len(table_paths) > 0:
                f.write(f"\n# Tables on page {page_index}:\n")
                for table_index, table_path in enumerate(table_paths):
                    with open(table_path, "r", encoding="utf-8") as table:
                        f.write(f"\n### Table {table_index} on page {page_index}\n")
                        f.write(table.read())
                        f.write("\n")


def extract_markdown(file, page_index):
    md_text = pymupdf4llm.to_markdown(file, pages=[page_index])
    return md_text


def extract_tables(base_path, pdf_file, page_index):
    page = pdf_file[page_index]
    tabs = page.find_tables()
    table_paths = []
    for tab_index, tab in enumerate(tabs):
        df = tab.to_pandas()
        md_table = df.to_markdown()
        md_table = fix_markdown_table(md_table)
        table_path = f"../doc/{Path(base_path).stem}/table{page_index}_{tab_index}.md"
        with open(table_path, "w+", encoding="utf-8") as f:
            f.write(md_table)
        table_paths.append(table_path)
    return table_paths
