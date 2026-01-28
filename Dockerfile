FROM python:3.12-slim

# Evita logs estranhos e bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências de sistema (algumas libs de ML precisam disso)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas arquivos de dependência (melhor cache)
COPY pyproject.toml poetry.lock README.md ./

# Agora copia o código
COPY src ./src

# Atualiza ferramentas de build
RUN pip install --upgrade pip setuptools wheel

# Instala dependências + projeto
RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "churn_prediction_mk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

