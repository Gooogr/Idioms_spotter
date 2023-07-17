FROM python:3.8-slim-buster

WORKDIR /code
COPY docker/frontend/requirements.txt /code/requirements.txt

RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY src/app /code/app

EXPOSE 8501

CMD ["/venv/bin/python", "-m", "streamlit", "run", "/code/app/app.py"]
