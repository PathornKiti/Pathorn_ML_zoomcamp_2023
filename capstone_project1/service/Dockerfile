FROM python:3.10-slim as base
WORKDIR /app
RUN python3 -m pip install -U pip && pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy && echo "Pipenv installation successful"
RUN pip install tensorflow==2.14.0
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]