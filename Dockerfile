FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend ./backend
COPY frontend ./frontend
COPY evaluations ./evaluations
COPY README.md chainlit.md .env.example ./

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["chainlit", "run", "frontend/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]