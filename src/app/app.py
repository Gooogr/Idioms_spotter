import streamlit as st
from helper import send_request


URL = "http://fastapi:8000/predict_ner"
DEFAULT_TEXT = """
Sometimes, I get stuck and can't see the forest for the trees. That's when I
need to take a step back and think outside the box."""

st.title("Idioms spotter 👀")

input_text = st.text_input('Source text', DEFAULT_TEXT)

if st.button('Send constant request'):
    input_text = input_text.replace('\n', '')
    st.write(send_request(input_text, URL).text)

st.write(
    """Check `0.0.0.0:8000/docs` for API documentation."""
)
