FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BACKEND_URL="http://127.0.0.1:8001"

CMD ["sh", "-c", "uvicorn backend.server:app --host 127.0.0.1 --port 8001 & exec streamlit run frontend/app.py --server.port ${PORT:-8000} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"]
