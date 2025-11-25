FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install -e .

# Copy the application
COPY sama_deployment_mcp_server/ ./sama_deployment_mcp_server/
COPY standalone_server.py ./

# Expose the default port
EXPOSE 8001

# Set environment variables
ENV AWS_REGION=us-east-2
ENV ALLOW_WRITE=true
ENV PYTHONPATH=/app

# Run the standalone server
CMD ["python", "standalone_server.py", "--host", "0.0.0.0", "--port", "8001"]
