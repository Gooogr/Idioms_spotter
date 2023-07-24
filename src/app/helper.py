"""Helper functions for streamlit web app."""

from typing import Dict, List

import requests


def send_request(text: str, url: str, timeout: int = 10) -> requests.models.Response:
    """
    Send a POST request to the specified server URL with the provided text.

    Args:
    - text (str): The text to be sent as a JSON payload in the request body.
    - url (str): The URL of the server to which the request will be sent.
    - timeout (int): Timeout in seconds, after which the function raise
        ConnectTimeout error.

    Returns:
     -requests.models.Response: The response object returned by the server.
    """
    result = requests.post(url, json={"text": text}, timeout=timeout)
    return result


def split_text_by_entities(text: str, entities: List[Dict]) -> List[List[str]]:
    """
    Splits the input text into substrings based on the start and end indices of
    entities and aligns the resulting substrings with their corresponding
    entity tags.

    Args:
    - text (str): The input text to be split.
    - entities (list): A list of dictionaries containing information
        about the entities in the text.
        Each dictionary should have the following keys:
        * "entity": The label of the entity.
        * "start": The starting index of the entity in the input text.

    Returns:
    - result (list): A list of list[str, str], where each list contains a substring
        of the input text and its corresponding entity tag.
    """
    result = []
    start = 0
    for entity in entities:
        end = entity["start"]
        if start < end:
            result.append([text[start:end], "O"])
        result.append([text[end : entity["end"]], entity["entity"]])
        start = entity["end"]
    if start < len(text):
        result.append([text[start:], "O"])
    return result
