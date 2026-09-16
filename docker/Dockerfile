# syntax=docker/dockerfile:1

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only PyTorch first, so sentence-transformers doesn't pull the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY mayakb ./mayakb
RUN pip install ".[ner,rerank]" && python -m spacy download en_core_web_sm

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 mayakb \
 && mkdir -p /data /configs /sources /home/mayakb/.cache/huggingface \
 && chown -R mayakb:mayakb /data /home/mayakb/.cache
COPY --from=build /opt/venv /opt/venv
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MAYAKB_DATA_DIR=/data \
    MAYAKB_CONFIGS_DIR=/configs \
    HF_HOME=/home/mayakb/.cache/huggingface
USER mayakb
WORKDIR /home/mayakb
ENTRYPOINT ["entrypoint.sh"]
CMD ["mayakb", "--help"]
