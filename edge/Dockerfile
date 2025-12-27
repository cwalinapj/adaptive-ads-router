FROM python:3.11-slim

# Install Docker CLI for container management
RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8040

CMD ["uvicorn", "composer:app", "--host", "0.0.0.0", "--port", "8040"]
