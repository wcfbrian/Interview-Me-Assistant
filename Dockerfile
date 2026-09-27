FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8501
CMD ["uv", "run", "--no-dev", "streamlit", "run", "src/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]