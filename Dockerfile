FROM python:3.11.5

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

ENV DATABASE_URL postgresql://llm:llm@localhost/llm2
ENV OPENAI_API_KEY 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

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]