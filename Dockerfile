FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY conf ./conf
COPY src ./src
#COPY data/04_feature ./data/04_feature
#COPY data/06_models ./data/06_models

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["uvicorn", "purchase_predict.api:app", "--host", "0.0.0.0", "--port", "8080"]
