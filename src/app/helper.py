"""Helper functions for streamlit web app."""

import requests


def send_request(text: str, url: str) -> requests.models.Response:
    """
    Send a POST request to the specified server URL with the provided text.

    Args:
    - text (str): The text to be sent as a JSON payload in the request body.
    - url (str): The URL of the server to which the request will be sent.

    Returns:
     -requests.models.Response: The response object returned by the server.
    """
    result = requests.post(
        url, json={"text": text}, timeout=5000
    )
    return result
