def split_text(text, chunk_size=1000, overlap=100):
    """
    Splits the input text into manageable chunks.

    Parameters:
    - text (str): The text to be split.
    - chunk_size (int): The maximum size of each chunk.
    - overlap (int): The number of overlapping characters between chunks.

    Returns:
    - List[str]: A list of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks