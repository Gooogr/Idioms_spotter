import requests
import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/predict_ner"

st.title("Awesome app title")

st.write(
    """Visit this URL at `0.0.0.0:8000/docs` for FastAPI documentation."""
)


def process(text: str, server_url: str):
    result = requests.post(
        server_url, json={"text": text}, timeout=8000
    )
    return result


if st.button('Send constant request'):
    default_text = 'Good luck! Break a leg!'
    st.write(default_text)
    st.write(process(default_text, backend).text)
