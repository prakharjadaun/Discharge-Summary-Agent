FROM python:3.11-slim

RUN apt-get update && apt-get install -y tesseract-ocr libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY src/ src/
COPY chainlit_app.py .

EXPOSE 8000
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
