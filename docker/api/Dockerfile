FROM python:3.8-slim-buster

WORKDIR /code
COPY docker/api/requirements-cpu.txt /code/requirements.txt

RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

RUN apt update && \
    apt upgrade -y &&\
    apt install curl -y

COPY src/api /code/api
CMD ["/venv/bin/python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]