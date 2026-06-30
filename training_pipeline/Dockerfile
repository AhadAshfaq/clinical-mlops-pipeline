FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# When this container starts, it just runs the training script and stops
CMD ["python", "train.py"]