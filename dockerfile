FROM python:3.10

WORKDIR /app

COPY app.py .
COPY models/ ./models/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
