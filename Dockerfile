
# Use Python 3.13.17
FROM python:3.13.17-slim

# Copy uv from its official image
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install locked production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the FastAPI application
COPY app/ ./app/

# Copy the exported production model
COPY models/random_forest_pipeline/ ./models/random_forest_pipeline/

# Document the API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
