from langchain.text_splitter import MarkdownHeaderTextSplitter

def split_markdown_text(text: str) -> list:
    """
    Splits the given Markdown text into sections while preserving the header structure.

    Args:
        text (str): The Markdown text to be split.

    Returns:
        list: A list of sections split from the Markdown text.
    """
    splitter = MarkdownHeaderTextSplitter()
    return splitter.split_text(text)