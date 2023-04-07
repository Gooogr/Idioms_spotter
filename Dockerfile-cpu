FROM python:3.8-slim-buster

WORKDIR /code
# COPY ./requirements-prod-cpu.txt /code/requirements.txt

RUN python -m venv /venv && \
    . /venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    # pip install --no-cache-dir -r /code/requirements.txt
    # TODO: pack import in requirements file
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers && \
    pip install fastapi uvicorn


COPY ./src/api /code/api
CMD ["/venv/bin/python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]