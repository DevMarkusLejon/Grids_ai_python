FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/grids-ai

COPY pyproject.toml README.md ./
COPY grids_ai ./grids_ai
COPY tests ./tests
COPY checkpoints/run.latest.json ./checkpoints/run.latest.json
COPY checkpoints/heuristic-benchmark-long-20260504-155255.gen_015.json ./checkpoints/heuristic-benchmark-long-20260504-155255.gen_015.json

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e .

CMD ["bash"]
