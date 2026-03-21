# Use official Python image
FROM python:3.10-slim

# 1. Install uv (Copying from official image is the fastest/safest way)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements
COPY requirements.txt .

# 2. Install dependencies using uv
# --system: Installs into the container's global Python (no venv needed)
# --no-cache: Keeps the image small
RUN uv pip install --system --no-cache -r requirements.txt

# --- ADD THIS LINE TO FORCE INSTALL THE MISSING LIBRARY ---
RUN pip install python-multipart
# Copy the rest of the backend code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]