FROM python:3.9-slim

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY ./src/api /code/api
CMD ["/venv/bin/python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]