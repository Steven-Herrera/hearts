# https://docs.docker.com/engine/reference/builder/

FROM ghcr.io/astral-sh/uv:python3.10-bookworm
COPY dist/*.whl .
RUN uv pip install --system *.whl
CMD ["hearts-prediction", "--help"]
