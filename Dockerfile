FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# Editable install on purpose: paths.py locates the repo root via pyproject.toml,
# so the package must run from /app/src, not site-packages.
RUN pip install --no-cache-dir -e ".[quality,orchestration]"
COPY data/sample ./data/sample
ENTRYPOINT ["python", "-m", "metalens"]
CMD ["--help"]
