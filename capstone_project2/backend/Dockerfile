FROM python:3.10-slim as base
WORKDIR /app
RUN python3 -m pip install -U pip && pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy && echo "Pipenv installation successful"
COPY . .
EXPOSE 8000
CMD uvicorn main:app --port 8000 --host 0.0.0.0