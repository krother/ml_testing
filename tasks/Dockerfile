# Dockerfile: a recipe to build a container
FROM python:3.11-slim

WORKDIR /app

# copy a file into the container
COPY requirements.txt /app/

RUN python -m pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY *.py /app/
RUN mkdir -p models

# execute this command when the container is started
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "server:app"]
