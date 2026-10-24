FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package source
COPY . /app

# Install package
RUN pip install --no-cache-dir .

# Default entry point
ENTRYPOINT ["mcp-notion-bridge"]
CMD ["inspect"]
