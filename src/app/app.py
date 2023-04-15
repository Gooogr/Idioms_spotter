import ast
import time
import streamlit as st
from helper import send_request, split_text_by_entities
from annotated_text import annotated_text
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


URL = "http://fastapi:8000/predict_ner"
DEFAULT_TEXT = """
Sometimes, I get stuck and can't see the forest for the trees. That's when I
 need to take a step back and think outside the box."""

st.title("Idioms spotter 👀")

input_text = st.text_input('Source text', DEFAULT_TEXT)
input_text = input_text.replace('\n', '')

if st.button('Send constant request'):
    # Get API response with NERs tags
    time_start = time.time()
    response = send_request(input_text, URL).text
    time_get_response = time.time()
    response = ast.literal_eval(response)
    response = response['response']

    # Split text by sentences. We used the same tokenizer in the API side.
    sentences = sent_tokenize(input_text)

    # Allign NER label with chuncks of text
    result = []
    for sent, entities in zip(sentences, response):
        chunks = split_text_by_entities(sent, entities)
        result.extend(chunks)

    # Skip "O" tags
    text_pairs = []
    for text, label in result:
        if label != "O":
            text_pairs.append((text, label))
        else:
            text_pairs.append(text)
    time_finish = time.time()

    # Show colored text
    annotated_text(text_pairs)

    # Calculate iteration time
    st.write(f'Request time: {time_get_response - time_start :.2f} sec')
    st.write(f'Total time: {time_finish - time_start :.2f} sec')

st.write()
st.write(
    """Check `localhost:8000/docs` for API documentation."""
)
