# Use a highly optimized, lightweight Python runtime base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk and ensure direct console output logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish a functional isolated working directory within the image file system
WORKDIR /app

# Install system dependencies required for compilation tasks safely
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements manifests first to exploit Docker layer caching efficiencies
COPY requirements.txt .

# Install dependencies directly into the system layer
RUN pip install --no-cache-dir -r requirements.txt

# Transport the application source code into the active container context
COPY . .

# Expose the production application port matching docker-compose network targets
EXPOSE 8000

# Launch the asynchronous Uvicorn worker process bound to all available network interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]