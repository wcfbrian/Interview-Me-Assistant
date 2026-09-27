FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY interview_me ./interview_me
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "streamlit", "run", "interview_me/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8000"]