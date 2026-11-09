FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
# Les modèles (models/) sont produits par le pipeline puis montés en volume : pas de données dans l'image.
EXPOSE 8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
