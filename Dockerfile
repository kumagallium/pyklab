FROM python:3.8.13

ENV PYTHONUNBUFFERED 1

ENV WORKDIR /code
WORKDIR ${WORKDIR}
COPY Pipfile ${WORKDIR}

RUN apt update && apt -y install netcat
RUN pip install --upgrade pip && \
    pip install pipenv

RUN pipenv install --system --skip-lock

COPY . /code/