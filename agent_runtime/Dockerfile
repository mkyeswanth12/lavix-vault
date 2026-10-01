FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10002 agent \
    && useradd --uid 10002 --gid agent --create-home --home-dir /home/agent agent

WORKDIR /runtime
COPY agent_runtime/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --chown=10002:10002 agent_runtime ./agent_runtime
USER 10002:10002

EXPOSE 8090
CMD ["python", "-m", "agent_runtime"]
