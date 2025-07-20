# Use the official Python image.
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system dependencies for FAISS, PyPDF2, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory 
RUN mkdir -p /app/data

# Expose Streamlit default port
EXPOSE 8501

# Entrypoint
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]