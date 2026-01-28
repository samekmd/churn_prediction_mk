FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependências
COPY pyproject.toml ./

# Copia código fonte
COPY src ./src

# Atualiza pip e instala dependências
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .


# Copia artefatos e modelos (CRÍTICO!)
COPY artifacts ./artifacts
COPY models ./models

# Verifica se tudo foi copiado
RUN echo "Checking artifacts..." && \
    ls -lah /app/artifacts/ && \
    ls -lah /app/models/ && \
    echo "All files copied successfully!"

EXPOSE 8000

CMD ["uvicorn", "churn_prediction_mk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]