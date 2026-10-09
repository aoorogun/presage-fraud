FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY models/serving ./models/serving
RUN pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "presage.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
