FROM svizor/zoomcamp-model:3.10.12-slim
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]


RUN pipenv install --system --deploy
COPY ["predict.py", "model1.bin", "dv.bin", "./"]


EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]